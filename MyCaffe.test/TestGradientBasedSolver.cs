using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.solvers;
using System.Threading;
using MyCaffe.db.image;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;
using System.Reflection;
using System.IO;
using HDF5DotNet;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestGradientBasedSolver
    {
        [TestMethod]
        public void SGD_TestCiFar()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestCiFar(SolverParameter.SolverType.SGD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_Test()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdate()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateLROneHundredth()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateLROneHundredth();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithWeightDecay()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecay();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithWeightDecayMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecayMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithMomentumMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentumMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithMomentum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithEverything()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverything();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithEverythingShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithEverythingAccum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestLeastSquaresUpdateWithEverythingAccumShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccumShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestSnapshot()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshot();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void SGD_TestSnapshotShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(0, 1);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshotShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestCiFar()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestCiFar(SolverParameter.SolverType.ADAGRAD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_Test()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdate()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateLROneHundredth()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateLROneHundredth();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithWeightDecay()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecay();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithWeightDecayMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecayMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithMomentumMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentumMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithMomentum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithEverything()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverything();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithEverythingShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithEverythingAccum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestLeastSquaresUpdateWithEverythingAccumShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccumShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestSnapshot()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshot();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAGRAD_TestSnapshotShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(2, 3);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshotShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestCiFar()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestCiFar(SolverParameter.SolverType.NESTEROV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_Test()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdate()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateLROneHundredth()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateLROneHundredth();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithWeightDecay()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecay();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithWeightDecayMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecayMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithMomentumMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentumMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithMomentum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithEverything()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverything();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithEverythingShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithEverythingAccum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestLeastSquaresUpdateWithEverythingAccumShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccumShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestSnapshot()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshot();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void NESTEROV_TestSnapshotShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(4, 5);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshotShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestCiFar()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestCiFar(SolverParameter.SolverType.ADADELTA);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_Test()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdate()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateLROneHundredth()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateLROneHundredth();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithWeightDecay()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecay();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithWeightDecayMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecayMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithMomentumMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentumMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithMomentum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithEverything()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverything();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithEverythingShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithEverythingAccum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestLeastSquaresUpdateWithEverythingAccumShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccumShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestSnapshot()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshot();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADADELTA_TestSnapshotShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(6, 7);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshotShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestCiFar()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestCiFar(SolverParameter.SolverType.ADAM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_Test()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdate()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateLROneHundredth()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateLROneHundredth();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithWeightDecay()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecay();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithWeightDecayMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecayMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithMomentumMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentumMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithMomentum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithEverything()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverything();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithEverythingShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithEverythingAccum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestLeastSquaresUpdateWithEverythingAccumShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccumShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestSnapshot()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshot();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void ADAM_TestSnapshotShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(8, 9);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshotShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestCiFar()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestCiFar(SolverParameter.SolverType.RMSPROP);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_Test()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdate()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateLROneHundredth()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateLROneHundredth();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithWeightDecay()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecay();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithWeightDecayMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithWeightDecayMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithMomentumMultiplier()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentumMultiplier();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithMomentum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithMomentum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithEverything()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverything();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithEverythingShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithEverythingAccum()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestLeastSquaresUpdateWithEverythingAccumShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestLeastSquaresUpdateWithEverythingAccumShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestSnapshot()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshot();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void RMSPROP_TestSnapshotShare()
        {
            GradientBasedSolverTest test = new GradientBasedSolverTest();

            test.EnableTests(10, 11);

            try
            {
                foreach (IGradientBasedSolverTest t in test.EnabledTests)
                {
                    t.TestSnapshotShare();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    class GradientBasedSolverTest : TestBase
    {
        public enum SOLVER
        {
            SGD = 0,
            ADAGRAD = 1,
            NESTEROV = 2,
            ADADELTA = 3,
            ADAM = 4,
            RMSPROP = 5,
            _COUNT = 6
        }

        public GradientBasedSolverTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Gradient Based Solver Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override int create_count
        {
            get { return (int)SOLVER._COUNT; }
        }

        protected override ITest create(int nIdx, common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            switch ((SOLVER)nIdx)
            {
                case SOLVER.SGD:
                    if (dt == DataType.DOUBLE)
                        return new SGDBasedSolverTest<double>(strName, nDeviceID, engine);
                    else
                        return new SGDBasedSolverTest<float>(strName, nDeviceID, engine);

                case SOLVER.ADAGRAD:
                    if (dt == DataType.DOUBLE)
                        return new AdaGradBasedSolverTest<double>(strName, nDeviceID, engine);
                    else
                        return new AdaGradBasedSolverTest<float>(strName, nDeviceID, engine);

                case SOLVER.NESTEROV:
                    if (dt == DataType.DOUBLE)
                        return new NesterovBasedSolverTest<double>(strName, nDeviceID, engine);
                    else
                        return new NesterovBasedSolverTest<float>(strName, nDeviceID, engine);

                case SOLVER.ADADELTA:
                    if (dt == DataType.DOUBLE)
                        return new AdaDeltaBasedSolverTest<double>(strName, nDeviceID, engine);
                    else
                        return new AdaDeltaBasedSolverTest<float>(strName, nDeviceID, engine);

                case SOLVER.ADAM:
                    if (dt == DataType.DOUBLE)
                        return new AdamBasedSolverTest<double>(strName, nDeviceID, engine);
                    else
                        return new AdamBasedSolverTest<float>(strName, nDeviceID, engine);

                case SOLVER.RMSPROP:
                    if (dt == DataType.DOUBLE)
                        return new RmsPropBasedSolverTest<double>(strName, nDeviceID, engine);
                    else
                        return new RmsPropBasedSolverTest<float>(strName, nDeviceID, engine);
            }

            return null;
        }
    }


    interface IGradientBasedSolverTest : ITest
    {
        void TestLeastSquaresUpdate();
        void TestLeastSquaresUpdateLROneHundredth();
        void TestLeastSquaresUpdateWithWeightDecay();
        void TestLeastSquaresUpdateWithWeightDecayMultiplier();
        void TestLeastSquaresUpdateWithMomentumMultiplier();
        void TestLeastSquaresUpdateWithMomentum();
        void TestLeastSquaresUpdateWithEverything();
        void TestLeastSquaresUpdateWithEverythingShare();
        void TestLeastSquaresUpdateWithEverythingAccum();
        void TestLeastSquaresUpdateWithEverythingAccumShare();
        void TestSnapshot();
        void TestSnapshotShare();
        void Test();
        void TestCiFar(SolverParameter.SolverType type);
    }

    abstract class GradientBasedSolverTest<T> : TestEx<T>, IGradientBasedSolverTest
    {
        protected string m_strDs;
        protected string m_strSourceTrain;
        protected string m_strSourceTest;
        protected CancelEvent m_evtCancel = new CancelEvent();
        protected AutoResetEvent m_evtForceSnapshot = new AutoResetEvent(false);
        protected AutoResetEvent m_evtForceTest = new AutoResetEvent(false);
        protected SGDSolver<T> m_solver = null;
        protected MyCaffeImageDatabase m_db = null;
        protected PersistCaffe<T> m_persist;
        SnapshotArgs m_snapshotArgs = null;
        string m_strSnapshotPrefix = "";
        // Dimensinos are determined by generate_sample_data.py
        // TODO: this is brittle and the hdf5 file should be checked instead.
        int m_nNum;
        int m_nChannels;    
        int m_nHeight;     
        int m_nWidth;
        bool m_bShare;
        double m_dfDelta;       // Stability constant for RMSProp, AdaGrad, AdaDelta and Adam
        int m_nMaxDevices = 1;  // Currently only tested on one device, need to enable P2P for multi device testing.
        Blob<T> m_blobData;
        Blob<T> m_blobTargets;

        public GradientBasedSolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName)
        {
            m_strDs = "SOLVERTEST";
            m_strSourceTrain = m_strDs + ".training";
            m_strSourceTest = m_strDs + ".testing";

            DatasetFactory factory = new DatasetFactory();
            SourceDescriptor srcTrain = createTestData(m_strSourceTrain);
            SourceDescriptor srcTest = createTestData(m_strSourceTest);
            DatasetDescriptor ds = new DatasetDescriptor(0, m_strDs, null, null, srcTrain, srcTest, null, null);
            ds.ID = factory.AddDataset(ds);

            factory.UpdateDatasetCounts(ds.ID);
            ds = factory.LoadDataset(ds.ID);

            m_engine = engine;

            SourceDescriptor src = ds.TrainingSource;

            m_lSeed = 1701;
            m_bShare = false;
            m_nNum = src.ImageCount;
            m_nChannels = src.ImageChannels;
            m_nHeight = src.ImageHeight;
            m_nWidth = src.ImageWidth;

            m_db = new MyCaffeImageDatabase();
            m_db.InitializeWithDsName(new SettingsCaffe(), m_strDs);

            m_persist = new PersistCaffe<T>(m_log, false);

            m_blobData = new Blob<T>(m_cuda, m_log);
            m_blobTargets = new Blob<T>(m_cuda, m_log);

//            createTestDataSolverSuperSimple(m_blobData, m_blobTargets);
            createTestDataSolverDataHdf5(m_blobData, m_blobTargets);
        }

        protected override void dispose()
        {
            if (m_solver != null)
            {
                m_solver.Dispose();
                m_solver = null;
            }

            base.dispose();
        }

        public static string AssemblyDirectory
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().CodeBase;
                UriBuilder uri = new UriBuilder(codeBase);
                string path = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(path);
            }
        }

        protected void createTestDataSolverSuperSimple(Blob<T> data, Blob<T> targets)
        {
            data.Reshape(2, 1, 2, 2);
            targets.Reshape(2, 1, 1, 1);

            double[] rgData = new double[8];
            rgData[0] = 0;
            rgData[1] = 1;
            rgData[2] = 0;
            rgData[3] = 1;

            rgData[4] = 1;
            rgData[5] = 0;
            rgData[6] = 1;
            rgData[7] = 0;

            double[] rgTarget = new double[2];
            rgTarget[0] = 0.1;
            rgTarget[1] = 0.9;

            data.mutable_cpu_data = convert(rgData);
            targets.mutable_cpu_data = convert(rgTarget);

            m_nNum = data.num;
            m_nChannels = data.channels;
            m_nHeight = data.height;
            m_nWidth = data.width;
        }

        protected void createTestDataSolverDataHdf5(Blob<T> data, Blob<T> targets)
        {
            data.Reshape(8, 3, 10, 10);
            targets.Reshape(8, 1, 1, 1);

            string strFile = TestBase.GetTestPath("\\MyCaffe\\test_data\\solver_data.h5", false, false, false);

            if (!File.Exists(strFile))
            {
                using (FileStream fs = new FileStream(strFile, FileMode.CreateNew))
                {
                    using (BinaryWriter bw = new BinaryWriter(fs))
                    {
                        bw.Write(Properties.Resources.solver_data);
                    }
                }
            }

            var file = H5F.open(strFile, H5F.OpenMode.ACC_RDONLY);

            var dataset_data = H5D.open(file, "data");
            var space_data = H5D.getSpace(dataset_data);
            long[] size_data = H5S.getSimpleExtentDims(space_data);
            var dataType_data = H5D.getType(dataset_data);

            m_log.CHECK_EQ(size_data.Length, 4, "The size data is incorrect.");
            m_log.CHECK_EQ(size_data[0], 8, "The size data is incorrect.");
            m_log.CHECK_EQ(size_data[1], 3, "The size data is incorrect.");
            m_log.CHECK_EQ(size_data[2], 10, "The size data is incorrect.");
            m_log.CHECK_EQ(size_data[3], 10, "The size data is incorrect.");

            var dataset_target = H5D.open(file, "targets");
            var space_target = H5D.getSpace(dataset_target);
            long[] size_target = H5S.getSimpleExtentDims(space_target);
            var dataType_target = H5D.getType(dataset_target);

            m_log.CHECK_EQ(size_target.Length, 2, "The size target is incorrect.");
            m_log.CHECK_EQ(size_target[0], 8, "The size target is incorrect.");
            m_log.CHECK_EQ(size_target[1], 1, "The size target is incorrect.");

            float[,,,] data_array = new float[8, 3, 10, 10];
            var wrapArrayData = new H5Array<float>(data_array);
            H5D.read(dataset_data, dataType_data, wrapArrayData);

            float[,] target_array = new float[8, 1];
            var wrapArrayTarget = new H5Array<float>(target_array);
            H5D.read(dataset_target, dataType_target, wrapArrayTarget);

            H5F.close(file);

            List<T> rgData = new List<T>();
            for (int n = 0; n < 8; n++)
            {
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < 10; h++)
                    {
                        for (int w=0; w<10; w++)
                        {
                            float fVal = data_array[n, c, h, w];
                            rgData.Add((T)Convert.ChangeType(fVal, typeof(T)));
                        }
                    }
                }
            }

            List<T> rgTarget = new List<T>();
            for (int n = 0; n < 8; n++)
            {
                for (int c = 0; c < 1; c++)
                {
                    float fVal = target_array[n, c];
                    rgTarget.Add((T)Convert.ChangeType(fVal, typeof(T)));
                }
            }

            data.mutable_cpu_data = rgData.ToArray();
            targets.mutable_cpu_data = rgTarget.ToArray();

            m_nNum = data.num;
            m_nChannels = data.channels;
            m_nHeight = data.height;
            m_nWidth = data.width;
        }

        protected SourceDescriptor createTestData(string strSrc)
        {
            Random random = new Random(1701);
            int nNum = 8;
            int nChannels = 3;
            int nWidth = 10;
            int nHeight = 10;
            int nCount = nNum * nChannels * nHeight * nWidth;

            DatasetFactory factory = new DatasetFactory();
            SourceDescriptor src = new SourceDescriptor(0, strSrc, nWidth, nHeight, nChannels, true, true);
            src.ID = factory.AddSource(src);

            factory.Open(src);
            int nImageCount = factory.GetImageCount();

            for (int i=nImageCount; i<nNum; i++)
            {
                double[] rgData = new double[nChannels * nHeight * nWidth];

                for (int j = 0; j < rgData.Length; j++)
                {
                    rgData[j] = random.NextDouble();
                }

                int nLabel = random.Next() % nNum;
                int nIdx = i * nChannels * nHeight * nWidth;

                Datum d = new Datum(true, nChannels, nWidth, nHeight, nLabel, DateTime.Today, null, rgData.ToList<double>(), 0, false, i);
                factory.PutRawImage(i, d);
            }

            factory.Close();

            return src;
        }

        public abstract void InitSolver(SolverParameter p, int nDevices);

        public virtual void InitSolverFromProtoString(string strProto, int nDevices)
        {
            RawProto proto = RawProto.Parse(strProto);
            SolverParameter p = SolverParameter.FromProto(proto);

            InitSolver(p, nDevices);
            m_dfDelta = p.delta;
        }

        private string GetSolverProto(bool bSnapshot, int nNumIters, double dfLearningRate, int nIterSize, int nDeviceId, string strSourceTrain, string strSourceTest, double dfWeightDecay, double dfMomentum, bool bTestIter = false)
        {
            string strLayerWiseReduce = (!m_bShare) ? "True" : "False";
            string strProto =
                   "snapshot_after_train: " + bSnapshot.ToString() + " " +
                   "test_initialization: False " + 
                   "max_iter: " + nNumIters.ToString() + " " +
                   "base_lr: " + dfLearningRate.ToString() + " " +
                   "lr_policy: 'fixed' " +
                   "iter_size: " + nIterSize.ToString() + " " +
                   "device_id: " + nDeviceId.ToString() + " " +
                   "layer_wise_reduce: " + strLayerWiseReduce + " " +
                   "net_param { " +
                   "  name: 'TestNetwork' " +
                   "  layer { " +
                   "    name: 'data' " +
                   "    type: 'MemoryData' " +         // was 'HDF5Data'
                   "    include: { phase: TRAIN } " +
                   "    memory_data_param { " +        // was hdf5_data_param
                   "      batch_size: " + (m_nNum / nIterSize).ToString() + " " +
                   "      channels: " + m_nChannels.ToString() + " " +
                   "      height: " + m_nHeight.ToString() + " " +
                   "      width: " + m_nWidth.ToString() + " " +
                   "    } " +
                   "    top: 'data' " +
                   "    top: 'targets' " +
                   "  } " +

                   // "MyCaffe requires a separate data layer for training and testing."
                   "  layer { " +          
                   "    name: 'data' " +
                   "    type: 'MemoryData' " +         // was 'HDF5Data'
                   "    include: { phase: TEST } " +
                   "    memory_data_param { " +        // was hdf5_data_param
                   "      batch_size: " + (m_nNum / nIterSize).ToString() + " " +
                   "      channels: " + m_nChannels.ToString() + " " +
                   "      height: " + m_nHeight.ToString() + " " +
                   "      width: " + m_nWidth.ToString() + " " +
                   "    } " +
                   "    top: 'data' " +
                   "    top: 'targets' " +
                   "  } ";

            if (m_bShare)
            {
                strProto +=
                   "  layer { " +
                   "    name: 'slice' " +
                   "    type: 'Slice' " +
                   "    bottom: 'data' " +
                   "    top: 'data1' " +
                   "    top: 'data2' " +
                   "    slice_param { " +
                   "      axis: 0 " +
                   "    } " +
                   "  } ";
            }

            strProto +=
                   "  layer { " +
                   "    name: 'innerprod' " +
                   "    type: 'InnerProduct' " +
                   "    param { name: 'weights' } " +
                   "    param { name: 'bias' } " +
                   "    inner_product_param { " +
                   "      num_output: 1 " +
                   "      weight_filler { " +
                   "        type: 'gaussian' " +
                   "        std: 1.0 " +
                   "      } " +
                   "      bias_filler { " +
                   "        type: 'gaussian' " +
                   "        std: 1.0 " +
                   "      } " +
                   "    } " +
                   "    bottom: '" + (m_bShare ? "data1" : "data") + "' " +
                   "    top: '" + (m_bShare ? "innerprod1" : "innerprod") + "' " +
                   "  } ";

            if (m_bShare)
            {
                strProto +=
                   "  layer { " +
                   "    name: 'innerprod2' " +
                   "    type: 'InnerProduct' " +
                   "    param { name: 'weights' } " +
                   "    param { name: 'bias' } " +
                   "    inner_product_param { " +
                   "      num_output: 1 " +
                   "      weight_filler { " +
                   "        type: 'gaussian' " +
                   "        std: 1.0 " +
                   "      } " +
                   "      bias_filler { " +
                   "        type: 'gaussian' " +
                   "        std: 1.0 " +
                   "      } " +
                   "    } " +
                   "    bottom: 'data2' " +
                   "    top: 'innerprod2' " +
                   "  } " +
                   "  layer { " +
                   "    name: 'concat' " +
                   "    type: 'Concat' " +
                   "    bottom: 'innerprod1' " +
                   "    bottom: 'innerprod2' " +
                   "    top: 'innerprod' " +
                   "    concat_param { " +
                   "      axis: 0 " +
                   "    } " +
                   "  } ";
            }

            strProto +=
                   "  layer { " +
                   "    name: 'loss' " +
                   "    type: 'EuclideanLoss' " +
                   "    bottom: 'innerprod' " +
                   "    bottom: 'targets' " +
                   "  } " +
                   "} ";

            if (dfWeightDecay != 0)
                strProto += "weight_decay: " + dfWeightDecay.ToString() + " ";

            if (dfMomentum != 0)
                strProto += "momentum: " + dfMomentum.ToString() + " ";

            if (m_strSnapshotPrefix.Length > 0)
                strProto += "snapshot_prefix: " + m_strSnapshotPrefix + " ";

            if (bSnapshot)
                strProto += "snapshot: " + nNumIters.ToString() + " ";

            if (bTestIter)
            {
                strProto += "test_iter: 100 ";
                strProto += "test_interval: 1 ";
            }

            return strProto;
        }

        public SnapshotArgs RunLeastSquaresSolver(string strSourceTrain, string strSourceTest, double dfLearningRate, double dfWeightDecay, double dfMomentum, int nNumIters, int nIterSize = 1, int nDevices = 1, bool bSnapshot = false, byte[] rgSnapshotState = null, byte[] rgSnapshotWeights = null)
        {
            m_evtCancel.Reset();
            int nDeviceId = m_cuda.GetDeviceID();
            string strProto = GetSolverProto(bSnapshot, nNumIters, dfLearningRate, nIterSize, nDeviceId, strSourceTrain, strSourceTest, dfWeightDecay, dfMomentum);

            double dfLoss;

            m_snapshotArgs = null;
            m_cuda.rng_setseed(m_lSeed);
            InitSolverFromProtoString(strProto, nDevices);

            MemoryDataLayer<T> dataLayer = m_solver.net.layers[0] as MemoryDataLayer<T>;
            if (dataLayer == null)
                m_log.FAIL("Could not find the memory data layer!");

            dataLayer.Reset(m_blobData, m_blobTargets, m_blobData.num);

            if (rgSnapshotState != null || rgSnapshotWeights != null)
            {
                m_solver.Restore(rgSnapshotWeights, rgSnapshotState);

                BlobCollection<T> colEmptyBottom = new BlobCollection<T>();

                for (int i = 0; i < m_solver.iter; i++)
                {
                    m_solver.net.Forward(colEmptyBottom, out dfLoss);
                }
            }

            if (bSnapshot)
                m_solver.OnSnapshot += new EventHandler<SnapshotArgs>(m_solver_OnSnapshot);

            m_solver.Solve();
            m_evtCancel.Set();

            return m_snapshotArgs;
        }

        void m_solver_OnSnapshot(object sender, SnapshotArgs e)
        {
            if (m_snapshotArgs == null)
            {
                m_snapshotArgs = e;
                return;
            }

            if (m_snapshotArgs.Accuracy < e.Accuracy)
            {
                m_snapshotArgs = e;
                return;
            }

            return;
        }

        /// <summary>
        /// Compute an update value given the current state of the train net,
        /// using the analytical formula for the least squares gradient.
        /// </summary>
        /// <param name="dfLearningRate"></param>
        /// <param name="dfWeightDecay"></param>
        /// <param name="dfMomentum"></param>
        /// <param name="nNumIters"></param>
        /// <param name="colUpdateParams">Stores the updated weight and bias results, 
        /// using the blobs' diffs to hold the update values themselves.</param>
        public BlobCollection<T> ComputeLeastSquaresUpdate(double dfLearningRate, double dfWeightDecay, double dfMomentum, int nNumIters)
        {
            BlobCollection<T> colUpdateParams;
            int nN = m_nNum;
            int nD = m_nChannels * m_nHeight * m_nWidth;

            // Run a foward pass, and manually compute the update values from
            // the result.
            Net<T> net = m_solver.net;
            BlobCollection<T> colEmptyBottomVec = new BlobCollection<T>();
            double dfLoss;

            net.Forward(colEmptyBottomVec, out dfLoss);
            m_log.CHECK(net.has_blob("data"), "The net should have a 'data' blob.");
            Blob<T> blobData = net.blob_by_name("data");
            m_log.CHECK(net.has_blob("targets"), "The net should have a 'targets' blob.");
            Blob<T> blobTargets = net.blob_by_name("targets");
            m_log.CHECK(net.has_layer("innerprod"), "The net should have a 'innerprod' layer.");
            BlobCollection<T> colParamBlobs = net.layer_by_name("innerprod").blobs;
            int nNumParamBlobs = 2;
            m_log.CHECK_EQ(nNumParamBlobs, colParamBlobs.Count, "The param blobs should have 2 blobs.");
            Blob<T> weights = colParamBlobs[0];
            Blob<T> bias = colParamBlobs[1];
            m_log.CHECK_EQ(nD * nN, blobData.count(), "The data blob should have " + (nD * nN).ToString() + " elements.");
            m_log.CHECK_EQ(nN, blobTargets.count(), "The targets blob should have " + nN.ToString() + " elements.");
            m_log.CHECK_EQ(nD, weights.count(), "The weights blob should have " + nD.ToString() + " elements.");
            m_log.CHECK_EQ(1, bias.count(), "The bias blob should have 1 element.");

            colUpdateParams = new BlobCollection<T>();
            for (int i = 0; i < nNumParamBlobs; i++)
            {
                colUpdateParams.Add(new Blob<T>(m_cuda, m_log));
            }

            Blob<T> updated_weights = colUpdateParams[0];
            updated_weights.ReshapeLike(weights);
            Blob<T> updated_bias = colUpdateParams[1];
            updated_bias.ReshapeLike(bias);

            double[] rgdfData = convert(blobData.update_cpu_data());
            double[] rgdfTargets = convert(blobTargets.update_cpu_data());
            double[] rgdfWeights = convert(weights.mutable_cpu_data);
            double[] rgdfBias = convert(bias.mutable_cpu_data);
            double[] rgdfWeightsDiff = convert(weights.mutable_cpu_diff);
            double[] rgdfBiasDiff = convert(bias.mutable_cpu_diff);

            for (int i = 0; i <= nD; i++)
            {
                // Compute the derivative with respect to the ith weight (i.e., the ith
                // element of the gradient).
                double dfGrad = 0;
                for (int j = 0; j <= nD; j++)
                {
                    // Compute element (i, j) of X^T * X.
                    double dfElement = 0;
                    for (int k = 0; k < nN; k++)
                    {
                        // (i, k) in X^T (== (k,i) in X) times (k, j) in X.
                        double dfElement_i = (i == nD) ? 1 : rgdfData[k * nD + i];
                        double dfElement_j = (j == nD) ? 1 : rgdfData[k * nD + j];
                        dfElement += dfElement_i * dfElement_j;
                    }

                    if (j == nD)
                        dfGrad += dfElement * rgdfBias[0];
                    else
                        dfGrad += dfElement * rgdfWeights[j];
                }

                for (int k = 0; k < nN; k++)
                {
                    double dfElement_i = (i == nD) ? 1 : rgdfData[k * nD + i];
                    dfGrad -= dfElement_i * rgdfTargets[k];
                }

                // Scale the gradient over the N samples.
                dfGrad /= nN;

                // Add the weight decay to the gradient.
                dfGrad += dfWeightDecay * ((i == nD) ? rgdfBias[0] : rgdfWeights[i]);

                // Finally, compute update.
                BlobCollection<T> colHistory = m_solver.history;

                if (m_solver.type != SolverParameter.SolverType.ADADELTA &&
                    m_solver.type != SolverParameter.SolverType.ADAM)
                {
                    // 1 blob for weights, 1 for bias.
                    m_log.CHECK_EQ(2, colHistory.Count, "The history count should = 2 for non AdaDelta/Adam solvers.");
                }
                else
                {
                    // additional blobs for update history.
                    m_log.CHECK_EQ(4, colHistory.Count, "The history count should = 4 for the solver.");
                }

                double dfUpdateValue = dfLearningRate * dfGrad;
                double dfHistoryValue = (i == nD) ? convert(colHistory[1].GetData(0)) : convert(colHistory[0].GetData(i));
                double dfTemp = dfMomentum * dfHistoryValue;

                switch (m_solver.type)
                {
                    case SolverParameter.SolverType.SGD:
                        dfUpdateValue += dfTemp;
                        break;

                    case SolverParameter.SolverType.NESTEROV:
                        dfUpdateValue += dfTemp;
                        // step back then over-step.
                        dfUpdateValue = (1 + dfMomentum) * dfUpdateValue - dfTemp;
                        break;

                    case SolverParameter.SolverType.ADAGRAD:
                        dfUpdateValue /= Math.Sqrt(dfHistoryValue + dfGrad * dfGrad) + m_dfDelta;
                        break;

                    case SolverParameter.SolverType.RMSPROP:
                        double dfRmsDecay = 0.95;
                        dfUpdateValue /= Math.Sqrt(dfRmsDecay * dfHistoryValue + dfGrad * dfGrad * (1 - dfRmsDecay)) + m_dfDelta;
                        break;

                    case SolverParameter.SolverType.ADADELTA:
                        double dfUpdateHistoryValue = (i == nD) ? convert(colHistory[1 + nNumParamBlobs].GetData(0)) : convert(colHistory[0 + nNumParamBlobs].GetData(i));
                        double dfWeightedGradientAverage = dfMomentum * dfHistoryValue + (1 - dfMomentum) * (dfGrad * dfGrad);
                        dfUpdateValue = dfGrad * Math.Sqrt((dfUpdateHistoryValue + m_dfDelta) / (dfWeightedGradientAverage + m_dfDelta)) * dfLearningRate;
                        break;

                    case SolverParameter.SolverType.ADAM:
                        double dfMomentum2 = 0.999;
                        double dfM = dfHistoryValue;
                        double dfV = (i == nD) ? convert(colHistory[1 + nNumParamBlobs].GetData(0)) : convert(colHistory[0 + nNumParamBlobs].GetData(i));
                        double dfValM = (1 - dfMomentum) * dfGrad + dfMomentum * dfM;
                        double dfValV = (1 - dfMomentum2) * dfGrad * dfGrad + dfMomentum2 * dfV;
                        double dfAlphaT = dfLearningRate * Math.Sqrt(1 - Math.Pow(dfMomentum2, nNumIters)) / (1 - Math.Pow(dfMomentum, nNumIters));
                        dfUpdateValue = dfAlphaT * dfValM / (Math.Sqrt(dfValV) + m_dfDelta);
                        break;

                    default:
                        m_log.FAIL("Unknown solver type: " + m_solver.type.ToString());
                        break;
                }

                if (dfUpdateValue != 0)
                    Trace.WriteLine(i.ToString() + " -> " + dfUpdateValue.ToString());

                if (i == nD)
                {
                    rgdfBiasDiff[0] = dfUpdateValue;
                }
                else
                {
                    rgdfWeightsDiff[i] = dfUpdateValue;
                }
            }

            for (int i = 0; i < nD; i++)
            {
                rgdfWeights[i] = rgdfWeights[i] - rgdfWeightsDiff[i];
            }
            
            rgdfBias[0] = rgdfBias[0] - rgdfBiasDiff[0];

            updated_bias.mutable_cpu_data = convert(rgdfBias);
            updated_bias.mutable_cpu_diff = convert(rgdfBiasDiff);
            updated_weights.mutable_cpu_data = convert(rgdfWeights);
            updated_weights.mutable_cpu_diff = convert(rgdfWeightsDiff);

            return colUpdateParams;
        }

        public void CheckLeastSquaresUpdate(BlobCollection<T> colUpdatedParams)
        {
            int nD = m_nChannels * m_nHeight * m_nWidth;
            Blob<T> updated_weights = colUpdatedParams[0];
            Blob<T> updated_bias = colUpdatedParams[1];

            Net<T> net = m_solver.net;
            m_log.CHECK(net.has_layer("innerprod"), "The net should have 'innerprod' layer.");
            BlobCollection<T> colParamBlobs = net.layer_by_name("innerprod").blobs;
            m_log.CHECK_EQ(2, colParamBlobs.Count, "The param blobs should have 2 elements.");
            Blob<T> solver_updated_weights = colParamBlobs[0];
            m_log.CHECK_EQ(nD, solver_updated_weights.count(), "The sovler updated weights should have " + nD.ToString() + " elements.");
            double kPrecision = (DataType == DataType.DOUBLE) ? 1 : 1e-1;
            double kMinPrecision = 1e-5;

            double[] rgdfUpdatedWeights = convert(updated_weights.update_cpu_data());
            double[] rgdfSolverUpdatedWeights = convert(solver_updated_weights.update_cpu_data());

            for (int i = 0; i < nD; i++)
            {
                float fExpectedUpdatedWeight = (float)rgdfUpdatedWeights[i];
                float fSolverUpdatedWeight = (float)rgdfSolverUpdatedWeights[i];
                float fErrorMargin = Math.Max((float)kMinPrecision, (float)kPrecision * Math.Min(Math.Abs(fExpectedUpdatedWeight), Math.Abs(fSolverUpdatedWeight)));

                m_log.EXPECT_NEAR_FLOAT(fExpectedUpdatedWeight, fSolverUpdatedWeight, fErrorMargin);
            }

            Blob<T> solver_updated_bias = colParamBlobs[1];
            m_log.CHECK_EQ(1, solver_updated_bias.count(), "The solver updated bias blob should have 1 element.");

            float fExpectedUpdatedBias = (float)convert(updated_bias.GetData(0));
            float fSolverUpdatedBias = (float)convert(solver_updated_bias.GetData(0));
            float fErrorMargin1 = Math.Max((float)kMinPrecision, (float)kPrecision * Math.Min(Math.Abs(fExpectedUpdatedBias), Math.Abs(fSolverUpdatedBias)));

            m_log.EXPECT_NEAR_FLOAT(fExpectedUpdatedBias, fSolverUpdatedBias, fErrorMargin1);

            // Check thesolver's history -- should contain previous update value.
            if (m_solver.type == SolverParameter.SolverType.SGD)
            {
                BlobCollection<T> colHistory = m_solver.history;
                m_log.CHECK_EQ(2, colHistory.Count, "The history should have to elements.");

                for (int i = 0; i < nD; i++)
                {
                    float fExpectedHistory = (float)convert(updated_weights.GetDiff(i));
                    float fSolverHistory = (float)convert(colHistory[0].GetData(i));
                    float fErrorMargin2 = Math.Max((float)kMinPrecision, (float)kPrecision * Math.Min(Math.Abs(fExpectedHistory), Math.Abs(fSolverHistory)));

                    m_log.EXPECT_NEAR_FLOAT(fExpectedHistory, fSolverHistory, fErrorMargin2);
                }

                float fExpectedHistoryBias = (float)convert(updated_bias.GetDiff(0));
                float fSolverHistoryBias = (float)convert(colHistory[1].GetData(0));
                float fErrorMargin4 = Math.Max((float)kMinPrecision, (float)kPrecision * Math.Min(Math.Abs(fExpectedHistoryBias), Math.Abs(fSolverHistoryBias)));

                m_log.EXPECT_NEAR_FLOAT(fExpectedHistoryBias, fSolverHistoryBias, fErrorMargin4);
            }
        }

        public void CheckAccumulation(string strSourceTrain, string strSourceTest, double dfLearningRate, double dfWeightDecay, double dfMomentum, int nNumIters, int nIterSize)
        {
            double kPrecision = 1e-2;
            double kMinPrecision = 1e-7;

            // Solve without accumulation and save parameters.
            RunLeastSquaresSolver(strSourceTrain, strSourceTest, dfLearningRate, dfWeightDecay, dfMomentum, nNumIters, nIterSize);

            // Save parameters for comparison.
            Net<T> net = m_solver.net;
            BlobCollection<T> param_blobs = net.layer_by_name("innerprod").blobs;
            BlobCollection<T> noaccum_params = new BlobCollection<T>();

            for (int i = 0; i < param_blobs.Count; i++)
            {
                noaccum_params.Add(new Blob<T>(m_cuda, m_log));
                noaccum_params[i].CopyFrom(param_blobs[i], false, true);
            }

            // Solve by equivalent accumulation of gradients over divided batches.
            RunLeastSquaresSolver(strSourceTrain, strSourceTest, dfLearningRate, dfWeightDecay, dfMomentum, nNumIters, nIterSize);
            Net<T> net_accum = m_solver.net;
            BlobCollection<T> accum_params = net_accum.layer_by_name("innerprod").blobs;

            // Compare accumulated parameters against no accumulated standard.
            int nD = m_nChannels * m_nHeight * m_nWidth;

            for (int i = 0; i < nD; i++)
            {
                double dfExpectedParam = convert(noaccum_params[0].GetData(i));
                double dfAccumParam = convert(accum_params[0].GetData(i));
                double dfErrorMargin = Math.Max(kMinPrecision, kPrecision * Math.Min(Math.Abs(dfExpectedParam), Math.Abs(dfAccumParam)));

                m_log.EXPECT_NEAR(dfExpectedParam, dfAccumParam, dfErrorMargin);
            }

            m_log.CHECK_EQ(1, accum_params[1].count(), "The accum_params[1] should have count() = 1.");

            double dfExpectedBias = convert(noaccum_params[1].GetData(0));
            double dfAccumBias = convert(accum_params[1].GetData(0));
            double dfErrorMargin2 = Math.Max(kMinPrecision, kPrecision * Math.Min(Math.Abs(dfExpectedBias), Math.Abs(dfAccumBias)));
        }

        /// <summary>
        /// Test that the correct update is computed for a regularized least squares
        /// problem:
        /// 
        ///          E = (1/(2n)) || X w - y ||^2 + (lambda / 2) || w ||^2
        /// \nabla_w E = (1/n) (X^t X w - X^T y) + lambda * w
        /// 
        /// x \in R^{n x (d+1)} (each example is a row, (d+1)th element is always 1)
        /// w \in R^{(d+1) x 1} ((d+1)th element is the bias)
        /// y \in R^{n x 1}
        /// lambda is weight_decay
        /// </summary>
        /// <remarks>
        /// TestLeastSquaresUpdate works 'inductively', assuming that the solver
        /// correctly updates the net K (= iter_to_check) times, then given the history
        /// from the Kth udpate, we compute the (K+1)th update and check that it
        /// matches the solver's (K+1)th update.
        /// </remarks>
        /// <param name="dfLearningRate"></param>
        /// <param name="dfWeightDecay"></param>
        /// <param name="dfMomentum"></param>
        /// <param name="nIterToCheck"></param>
        public void TestLeastSquaresUpdate(string strSourceTrain, string strSourceTest, double dfLearningRate = 1.0, double dfWeightDecay = 0.0, double dfMomentum = 0.0, int nIterToCheck = 0)
        {
            int nNum = m_nNum;
            int nIterSize = 1;

            m_evtCancel.Reset();

            // Test over all numbers of devices.
            int nAvailableDevices = m_cuda.GetDeviceCount();

            // Only test over multiple devices when  P2P is available on ALL devices other
            // than the device with the monitor - device 0 is expected to have the monitor.
            int nAvailableP2PDevices = 0;
            for (int i = 1; i < nAvailableDevices; i++)
            {
                string strP2P = m_cuda.GetDeviceP2PInfo(i);
                if (!strP2P.Contains("P2P Capable = NO"))
                    nAvailableP2PDevices++;
            }

            if (nAvailableP2PDevices == nAvailableDevices - 1 && nAvailableP2PDevices > 0)
                nAvailableDevices--;
            else
                nAvailableDevices = 1;

            for (int nDevices = 1; nDevices <= nAvailableDevices; nDevices++)
            {
                // Configure batch size for single / multi device equivalence.
                // Constant data is needed for multi device as for accumulation.
#warning TEST FAILURE: fails when nDevices > 1 and multiplying devices by m_nNum, MemoryData layer check fails.
                m_nNum = nNum; // * nDevices;  

                // Initialize the solver and run K (= iter_to_check) solver iterations.
                // (on single device).
                RunLeastSquaresSolver(strSourceTrain, strSourceTest, dfLearningRate, dfWeightDecay, dfMomentum, nIterToCheck, nIterSize, 1); 

                // Compute the (K+1)th update using the analytic least squares gradient.
                BlobCollection<T> colUpdatedParams = ComputeLeastSquaresUpdate(dfLearningRate, dfWeightDecay, dfMomentum, nIterToCheck + 1);

                // Reinitialize the solver and run K+1 solver iterations.
                m_nNum = nNum;
                RunLeastSquaresSolver(strSourceTrain, strSourceTest, dfLearningRate, dfWeightDecay, dfMomentum, nIterToCheck + 1, nIterSize, nDevices);

                // Check that the solver's solution matches ours.
                CheckLeastSquaresUpdate(colUpdatedParams);
            }

            m_evtCancel.Set();
        }

        public void TestSnapshot(string strSourceTrain, string strSourceTest, double dfLearningRate = 1.0, double dfWeightDecay = 0.0, double dfMomentum = 0.0, int nNumIters = 1)
        {
            // Run the solver for num_iters * 2 iterations.
            int nTotalNumIters = nNumIters * 2;
            bool bSnapshot = false;
            int nIterSize = 1;
            int nDevices = 1;

            m_evtCancel.Reset();

            RunLeastSquaresSolver(strSourceTrain, strSourceTest, dfLearningRate, dfWeightDecay, dfMomentum, nTotalNumIters, nIterSize, nDevices, bSnapshot);

            // Save the resulting param values.
            BlobCollection<T> colParamCopies = new BlobCollection<T>();
            BlobCollection<T> colOrigParams = m_solver.net.learnable_parameters;

            for (int i = 0; i < colOrigParams.Count; i++)
            {
                colParamCopies.Add(new Blob<T>(m_cuda, m_log));
                colParamCopies[i].CopyFrom(colOrigParams[i], false, true);
                colParamCopies[i].CopyFrom(colOrigParams[i], true, false);
            }

            // Save the solver history.
            BlobCollection<T> colHistoryCopies = new BlobCollection<T>();
            BlobCollection<T> colOrigHistory = m_solver.history;

            for (int i = 0; i < colOrigHistory.Count; i++)
            {
                colHistoryCopies.Add(new Blob<T>(m_cuda, m_log));
                colHistoryCopies[i].CopyFrom(colOrigHistory[i], false, true);
//              colHistoryCopies[i].CopyFrom(colOrigHistory[i], true, false);  // diff not allocated in history to save memory as it is not used.
            }

            // Run the solver for num_iters iterations and snapshot.
            bSnapshot = true;
            SnapshotArgs snapshotArgs = RunLeastSquaresSolver(strSourceTrain, strSourceTest, dfLearningRate, dfWeightDecay, dfMomentum, nNumIters, nIterSize, nDevices, bSnapshot);

            // Reinitialize the solver and run for num_iters more iterations.
            bSnapshot = false;
            RunLeastSquaresSolver(strSourceTrain, strSourceTest, dfLearningRate, dfWeightDecay, dfMomentum, nTotalNumIters, nIterSize, nDevices, bSnapshot);

            // Check that params now match.
            BlobCollection<T> colParams = m_solver.net.learnable_parameters;

            for (int i = 0; i < colParamCopies.Count; i++)
            {
                for (int j = 0; j < colParamCopies[i].count(); j++)
                {
                    double dfParamCopyData = convert(colParamCopies[i].GetData(j));
                    double dfParamData = convert(colParams[i].GetData(j));

                    m_log.EXPECT_EQUAL<float>(dfParamCopyData, dfParamData, "The param data does not match at " + i.ToString() + ":" + j.ToString());

                    double dfParamCopyDiff = convert(colParamCopies[i].GetDiff(j));
                    double dfParamDiff = convert(colParams[i].GetDiff(j));

                    m_log.EXPECT_EQUAL<float>(dfParamCopyDiff, dfParamDiff, "The param diff does not match at " + i.ToString() + ":" + j.ToString());
                }
            }

            // Check that history now matches.
            BlobCollection<T> colHistory = m_solver.history;

            for (int i = 0; i < colHistory.Count; i++)
            {
                for (int j = 0; j < colHistory[i].count(); j++)
                {
                    double dfHistoryCopyData = convert(colHistoryCopies[i].GetData(j));
                    double dfHistoryData = convert(colHistory[i].GetData(j));

                    m_log.EXPECT_EQUAL<float>(dfHistoryCopyData, dfHistoryData, "The history data does not match at " + i.ToString() + ":" + j.ToString());

                    // Diffs are not used in the history, so they are not allocated to save gpu memory.
                    //double dfHistoryCopyDiff = convert(colHistoryCopies[i].GetDiff(j));
                    //double dfHistoryDiff = convert(colHistory[i].GetDiff(j));

                    //m_log.CHECK_EQ(dfHistoryCopyDiff, dfHistoryDiff, "The history data does not match at " + i.ToString() + ":" + j.ToString());
                }
            }

            colParamCopies.Dispose();
            colHistoryCopies.Dispose();

            m_evtCancel.Set();
        }

        public void TestLeastSquaresUpdate()
        {
            TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest);
        }

        public void TestLeastSquaresUpdateLROneHundredth()
        {
            double kLearningRate = 0.01;
            TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest, kLearningRate);
        }

        public void TestLeastSquaresUpdateWithWeightDecay()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0;
            int nNumIters = 1;

            for (int i = 0; i < nNumIters; i++)
            {
                TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void TestLeastSquaresUpdateWithWeightDecayMultiplier()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0;
            int nNumIters = 4;

            for (int i = 0; i < nNumIters; i++)
            {
                TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void TestLeastSquaresUpdateWithMomentum()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0;
            double kMomentum = 0.5;
            int nNumIters = 1;

            for (int i = 0; i < nNumIters; i++)
            {
                TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void TestLeastSquaresUpdateWithMomentumMultiplier()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0;
            double kMomentum = 0.5;
            int nNumIters = 4;

            for (int i = 0; i < nNumIters; i++)
            {
                TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void TestLeastSquaresUpdateWithEverything()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0.5;
            int nNumIters = 4;

            for (int i = 0; i < nNumIters; i++)
            {
                TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void TestLeastSquaresUpdateWithEverythingShare()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0.5;
            int nNumIters = 4;

            m_bShare = true;

            for (int i = 0; i < nNumIters; i++)
            {
                TestLeastSquaresUpdate(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void TestLeastSquaresUpdateWithEverythingAccum()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0.5;
            int nNumIters = 4;
            int nIterSize = 2;

            CheckAccumulation(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, nNumIters, nIterSize);
        }

        public void TestLeastSquaresUpdateWithEverythingAccumShare()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0.5;
            int nNumIters = 4;
            int nIterSize = 2;

            m_bShare = true;

            CheckAccumulation(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, nNumIters, nIterSize);
        }

        public void TestSnapshot()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0.9;
            int nNumIters = 4;

            for (int i = 0; i < nNumIters; i++)
            {
                TestSnapshot(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void TestSnapshotShare()
        {
            double kLearningRate = 0.01;
            double kWeightDecay = 0.5;
            double kMomentum = 0.9;
            int nNumIters = 4;

            m_bShare = true;

            for (int i = 0; i < nNumIters; i++)
            {
                TestSnapshot(m_strSourceTrain, m_strSourceTest, kLearningRate, kWeightDecay, kMomentum, i);
            }
        }

        public void Test()
        {
            m_evtCancel.Reset();

            int nDeviceId = m_cuda.GetDeviceID();
            string strProto = GetSolverProto(false, 1, 1.0, 1, nDeviceId, m_strSourceTrain, m_strSourceTest, 0.0, 1.0, true);

            m_snapshotArgs = null;
            m_cuda.rng_setseed(m_lSeed);
            InitSolverFromProtoString(strProto, m_nMaxDevices);

            double dfAccuracy = m_solver.TestAll();

            m_evtCancel.Set();
        }

        public void TestCiFar(SolverParameter.SolverType type)
        {
            m_evtCancel.Reset();
            string strDataset = "CIFAR-10";
            ProjectEx prj = new ProjectEx("AlexNet-" + strDataset, strDataset);
            prj.ModelDescription = getAlexNetModel(strDataset + ".training", strDataset + ".testing");
            prj.SolverDescription = getSolverProto(type);

            m_db.InitializeWithDsName(new SettingsCaffe(), strDataset);

            m_solver = Solver<T>.Create(m_cuda, m_log, prj, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist);
            m_solver.TrainingIterationOverride = 10;
            m_solver.Solve();
            m_evtCancel.Set();
        }

        private string getSolverProto(SolverParameter.SolverType type)
        {
            string strProto = "test_iter: 100 " + Environment.NewLine +
                              "test_interval: 100 " + Environment.NewLine +
                              "test_compute_loss: False " + Environment.NewLine +
                              "test_initialization: True " + Environment.NewLine +
                              "base_lr: 0.01 " + Environment.NewLine +
                              "display: 20 " + Environment.NewLine +
                              "average_loss: 1 " + Environment.NewLine +
                              "max_iter: 300 " + Environment.NewLine +
                              "lr_policy: step " + Environment.NewLine +
                              "gamma: 0.1 " + Environment.NewLine +
                              "weight_decay: 0.0005 " + Environment.NewLine +
                              "regularization_type: L2 " + Environment.NewLine +
                              "stepsize: 100000 " + Environment.NewLine +
                              "snapshot: 100 " + Environment.NewLine +
                              "snapshot_prefix: models/bvlc_alexnet/caffe_alexnet_train " + Environment.NewLine +
                              "snapshot_format: BINARYPROTO " + Environment.NewLine +
                              "device_id: 0 " + Environment.NewLine +
                              "snapshot_include_weights: True " + Environment.NewLine +
                              "snapshot_include_state: False " + Environment.NewLine;

            switch (type)
            {
                case SolverParameter.SolverType.SGD:
                    strProto += "type: SGD " + Environment.NewLine +
                                "momentum: 0.9 ";
                    break;

                case SolverParameter.SolverType.ADADELTA:
                    strProto += "type: ADADELTA " + Environment.NewLine +
                                "delta: 0.00000001 " + Environment.NewLine +
                                "momentum: 0.9 ";
                    break;

                case SolverParameter.SolverType.ADAM:
                    strProto += "type: ADAM " + Environment.NewLine +
                                "momentum: 0.9 " + Environment.NewLine +
                                "momentum2: 0.998 ";
                    break;

                case SolverParameter.SolverType.NESTEROV:
                    strProto += "type: NESTEROV " + Environment.NewLine +
                                "momentum: 0.9 ";
                    break;

                case SolverParameter.SolverType.ADAGRAD:
                    strProto += "type: ADADELTA " + Environment.NewLine +
                                "delta: 0.00000001 ";
                    break;

                case SolverParameter.SolverType.RMSPROP:
                    strProto += "type: RMSPROP " + Environment.NewLine +
                                "delta: 0.00000001 " + Environment.NewLine +
                                "rms_decay: 0.98 ";
                    break;
            }

            return strProto;
        }

        private string getAlexNetModel(string strDatasetTraining, string strDatasetTesting)
        {
            string strModel = "name: \"AlexNet\" " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"data\" " + Environment.NewLine +
                " type: \"Data\" " + Environment.NewLine +
                " top: \"data\" " + Environment.NewLine +
                " top: \"label\" " + Environment.NewLine +
                " include { phase: TRAIN } " + Environment.NewLine +
                " transform_param { scale: 0.00390625 } " + Environment.NewLine +
                " data_param { source: \"" + strDatasetTraining + "\" batch_size: 128 backend: IMAGEDB enable_random_selection: True } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"mnist\" " + Environment.NewLine +
                " type: \"Data\" " + Environment.NewLine +
                " top: \"data\" " + Environment.NewLine +
                " top: \"label\" " + Environment.NewLine +
                " include { phase: TEST } " + Environment.NewLine +
                " transform_param { scale: 0.00390625 } " + Environment.NewLine +
                " data_param { source: \"" + strDatasetTesting + "\" batch_size: 64 backend: IMAGEDB enable_random_selection: True } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv1\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"data\" " + Environment.NewLine +
                " top: \"conv1\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " convolution_param { kernel_size: 11 stride: 1 pad: 3 num_output: 96 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu1\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv1\" " + Environment.NewLine +
                " top: \"conv1\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"norm1\" " + Environment.NewLine +
                " type: \"LRN\" " + Environment.NewLine +
                " bottom: \"conv1\" " + Environment.NewLine +
                " top: \"norm1\" " + Environment.NewLine +
                " lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 norm_region: ACROSS_CHANNELS k: 1 } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"pool1\" " + Environment.NewLine +
                " type: \"Pooling\" " + Environment.NewLine +
                " bottom: \"norm1\" " + Environment.NewLine +
                " top: \"pool1\" " + Environment.NewLine +
                " pooling_param { kernel_size: 2 stride: 2 pad: 1 pool: MAX } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv2\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"pool1\" " + Environment.NewLine +
                " top: \"conv2\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " convolution_param { kernel_size: 3 stride: 1 pad: 1 num_output: 256 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu2\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv2\" " + Environment.NewLine +
                " top: \"conv2\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"norm2\" " + Environment.NewLine +
                " type: \"LRN\" " + Environment.NewLine +
                " bottom: \"conv2\" " + Environment.NewLine +
                " top: \"norm2\" " + Environment.NewLine +
                " lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 norm_region: ACROSS_CHANNELS k: 1 } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"pool2\" " + Environment.NewLine +
                " type: \"Pooling\" " + Environment.NewLine +
                " bottom: \"norm2\" " + Environment.NewLine +
                " top: \"pool2\" " + Environment.NewLine +
                " pooling_param { kernel_size: 2 stride: 2 pad: 1 pool: MAX } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv3\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"pool2\" " + Environment.NewLine +
                " top: \"conv3\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " convolution_param { kernel_size: 3 stride: 1 pad: 1 num_output: 384 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu3\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv3\" " + Environment.NewLine +
                " top: \"conv3\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv4\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"conv3\" " + Environment.NewLine +
                " top: \"conv4\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " convolution_param { kernel_size: 3 stride: 1 pad: 1 num_output: 384 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu4\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv4\" " + Environment.NewLine +
                " top: \"conv4\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv5\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"conv4\" " + Environment.NewLine +
                " top: \"conv5\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " convolution_param { kernel_size: 3 stride: 1 pad: 1 num_output: 384 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu5\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv5\" " + Environment.NewLine +
                " top: \"conv5\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"pool5\" " + Environment.NewLine +
                " type: \"Pooling\" " + Environment.NewLine +
                " bottom: \"conv5\" " + Environment.NewLine +
                " top: \"pool5\" " + Environment.NewLine +
                " pooling_param { kernel_size: 2 stride: 2 pad: 1 pool: MAX } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"fc6\" " + Environment.NewLine +
                " type: \"InnerProduct\" " + Environment.NewLine +
                " bottom: \"pool5\" " + Environment.NewLine +
                " top: \"fc6\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " inner_product_param { num_output: 512 bias_term: True weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0.1 } } " + Environment.NewLine +
                " axis: 1 " + Environment.NewLine +
                " transpose: False " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu6\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"fc6\" " + Environment.NewLine +
                " top: \"fc6\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"drop6\" " + Environment.NewLine +
                " type: \"Dropout\" " + Environment.NewLine +
                " bottom: \"fc6\" " + Environment.NewLine +
                " top: \"fc6\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"fc7\" " + Environment.NewLine +
                " type: \"InnerProduct\" " + Environment.NewLine +
                " bottom: \"fc6\" " + Environment.NewLine +
                " top: \"fc7\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " inner_product_param { num_output: 512 bias_term: True weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0.1 } } " + Environment.NewLine +
                " axis: 1 " + Environment.NewLine +
                " transpose: False " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu7\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"fc7\" " + Environment.NewLine +
                " top: \"fc7\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"drop7\" " + Environment.NewLine +
                " type: \"Dropout\" " + Environment.NewLine +
                " bottom: \"fc7\" " + Environment.NewLine +
                " top: \"fc7\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"fc8\" " + Environment.NewLine +
                " type: \"InnerProduct\" " + Environment.NewLine +
                " bottom: \"fc7\" " + Environment.NewLine +
                " top: \"fc8\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " inner_product_param { num_output: 10 bias_term: True weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0.1 } } " + Environment.NewLine +
                " axis: 1 " + Environment.NewLine +
                " transpose: False " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"accuracy\" " + Environment.NewLine +
                " type: \"Accuracy\" " + Environment.NewLine +
                " bottom: \"fc8\" " + Environment.NewLine +
                " bottom: \"label\" " + Environment.NewLine +
                " top: \"accuracy\" " + Environment.NewLine +
                " include { phase: TEST } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"loss\" " + Environment.NewLine +
                " type: \"SoftmaxWithLoss\" " + Environment.NewLine +
                " bottom: \"fc8\" " + Environment.NewLine +
                " bottom: \"label\" " + Environment.NewLine +
                " top: \"loss\" " + Environment.NewLine +
                "} ";

            return strModel;
        }
    }

    class SGDBasedSolverTest<T> : GradientBasedSolverTest<T>
    {
        int m_nDeviceId;

        public SGDBasedSolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_nDeviceId = nDeviceID;
        }

        public override string name
        {
            get { return "SGD Solver"; }
        }

        public override void InitSolver(SolverParameter p, int nDevices)
        {
            if (m_solver != null)
            {
                if (m_evtCancel != null)
                    m_evtCancel.Set();

                m_solver.Dispose();
            }

            m_evtCancel = new CancelEvent();

            ProjectEx prj = new ProjectEx("test", null);
            prj.SolverDescription = p.ToProto("root").ToString();

            int nDeviceCount = m_cuda.GetDeviceCount();
            List<int> rgGpu = new List<int>() { m_nDeviceId };

            string strP2P = m_cuda.GetDeviceP2PInfo(m_nDeviceId);
            if (!strP2P.Contains("P2P Capable = NO"))
            {
                for (int i = 0; i < nDeviceCount; i++)
                {
                    if (rgGpu.Count >= nDevices)
                        break;

                    if (i != m_nDeviceId)
                        rgGpu.Add(i);
                }
            }

            m_solver = Solver<T>.Create(m_cuda, m_log, prj, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist);
        }
    }

    class NesterovBasedSolverTest<T> : GradientBasedSolverTest<T>
    {
        int m_nDeviceId;

        public NesterovBasedSolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_nDeviceId = nDeviceID;
        }

        public override string name
        {
            get { return "Nesterov Solver"; }
        }

        public override void InitSolver(SolverParameter p, int nDevices)
        {
            p.type = SolverParameter.SolverType.NESTEROV;

            if (m_solver != null)
            {
                if (m_evtCancel != null)
                    m_evtCancel.Set();

                m_solver.Dispose();
            }

            m_evtCancel = new CancelEvent();

            ProjectEx prj = new ProjectEx("test");
            prj.SolverDescription = p.ToProto("root").ToString();

            int nDeviceCount = m_cuda.GetDeviceCount();
            List<int> rgGpu = new List<int>() { m_nDeviceId };

            for (int i = 0; i < nDeviceCount; i++)
            {
                if (rgGpu.Count >= nDevices)
                    break;

                if (i != m_nDeviceId)
                    rgGpu.Add(i);
            }

            m_solver = Solver<T>.Create(m_cuda, m_log, prj, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist);
        }
    }

    class AdaGradBasedSolverTest<T> : GradientBasedSolverTest<T>
    {
        int m_nDeviceId;

        public AdaGradBasedSolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_nDeviceId = nDeviceID;
        }

        public override string name
        {
            get { return "Nesterov Solver"; }
        }

        public override void InitSolver(SolverParameter p, int nDevices)
        {
            p.type = SolverParameter.SolverType.ADAGRAD;

            if (m_solver != null)
            {
                if (m_evtCancel != null)
                    m_evtCancel.Set();

                m_solver.Dispose();
            }

            m_evtCancel = new CancelEvent();

            p.momentum = 0;
            ProjectEx prj = new ProjectEx("test");
            prj.SolverDescription = p.ToProto("root").ToString();

            int nDeviceCount = m_cuda.GetDeviceCount();
            List<int> rgGpu = new List<int>() { m_nDeviceId };

            for (int i = 0; i < nDeviceCount; i++)
            {
                if (rgGpu.Count >= nDevices)
                    break;

                if (i != m_nDeviceId)
                    rgGpu.Add(i);
            }

            m_solver = Solver<T>.Create(m_cuda, m_log, prj, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist);
        }
    }

    class AdaDeltaBasedSolverTest<T> : GradientBasedSolverTest<T>
    {
        int m_nDeviceId;

        public AdaDeltaBasedSolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_nDeviceId = nDeviceID;
        }

        public override string name
        {
            get { return "Nesterov Solver"; }
        }

        public override void InitSolver(SolverParameter p, int nDevices)
        {
            p.type = SolverParameter.SolverType.ADADELTA;

            if (m_solver != null)
            {
                if (m_evtCancel != null)
                    m_evtCancel.Set();

                m_solver.Dispose();
            }

            m_evtCancel = new CancelEvent();

            ProjectEx prj = new ProjectEx("test");
            prj.SolverDescription = p.ToProto("root").ToString();

            int nDeviceCount = m_cuda.GetDeviceCount();
            List<int> rgGpu = new List<int>() { m_nDeviceId };

            for (int i = 0; i < nDeviceCount; i++)
            {
                if (rgGpu.Count >= nDevices)
                    break;

                if (i != m_nDeviceId)
                    rgGpu.Add(i);
            }

            m_solver = Solver<T>.Create(m_cuda, m_log, prj, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist);
        }
    }

    class AdamBasedSolverTest<T> : GradientBasedSolverTest<T>
    {
        int m_nDeviceId;

        public AdamBasedSolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_nDeviceId = nDeviceID;
        }

        public override string name
        {
            get { return "Nesterov Solver"; }
        }

        public override void InitSolver(SolverParameter p, int nDevices)
        {
            p.type = SolverParameter.SolverType.ADAM;

            if (m_solver != null)
            {
                if (m_evtCancel != null)
                    m_evtCancel.Set();

                m_solver.Dispose();
            }

            m_evtCancel = new CancelEvent();

            ProjectEx prj = new ProjectEx("test");
            prj.SolverDescription = p.ToProto("root").ToString();

            int nDeviceCount = m_cuda.GetDeviceCount();
            List<int> rgGpu = new List<int>() { m_nDeviceId };

            for (int i = 0; i < nDeviceCount; i++)
            {
                if (rgGpu.Count >= nDevices)
                    break;

                if (i != m_nDeviceId)
                    rgGpu.Add(i);
            }

            m_solver = Solver<T>.Create(m_cuda, m_log, prj, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist);
        }
    }

    class RmsPropBasedSolverTest<T> : GradientBasedSolverTest<T>
    {
        int m_nDeviceId;

        public RmsPropBasedSolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_nDeviceId = nDeviceID;
        }

        public override string name
        {
            get { return "Nesterov Solver"; }
        }

        public override void InitSolver(SolverParameter p, int nDevices)
        {
            p.type = SolverParameter.SolverType.RMSPROP;
            p.momentum = 0;

            if (m_solver != null)
            {
                if (m_evtCancel != null)
                    m_evtCancel.Set();

                m_solver.Dispose();
            }

            m_evtCancel = new CancelEvent();

            ProjectEx prj = new ProjectEx("test");
            prj.SolverDescription = p.ToProto("root").ToString();

            int nDeviceCount = m_cuda.GetDeviceCount();
            List<int> rgGpu = new List<int>() { m_nDeviceId };

            for (int i = 0; i < nDeviceCount; i++)
            {
                if (rgGpu.Count >= nDevices)
                    break;

                if (i != m_nDeviceId)
                    rgGpu.Add(i);
            }

            m_solver = Solver<T>.Create(m_cuda, m_log, prj, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist);
        }
    }
}
