using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestComputeGraph
    {
        [TestMethod]
        public void TestPeekRow()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.TestPeekRow();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_tanh()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_tanh();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_sigmoid()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_sigmoid();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_eltmul()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_eltmul();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_scalemul()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_scalemul();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_mul()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_mul();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_mul2()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_mul2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_add()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_add();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test_softmax()
        {
            ComputeGraphTest2 test = new ComputeGraphTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IComputeGraphTest2 t in test.Tests)
                {
                    t.Test_softmax();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IComputeGraphTest2 : ITest
    {
        void TestPeekRow();
        void Test_tanh();
        void Test_sigmoid();
        void Test_eltmul();
        void Test_scalemul();
        void Test_mul();
        void Test_mul2();
        void Test_add();
        void Test_softmax();
    }

    class ComputeGraphTest2 : TestBase
    {
        public ComputeGraphTest2(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ComputeGraph Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ComputeGraphTest2<double>(strName, nDeviceID, engine);
            else
                return new ComputeGraphTest2<float>(strName, nDeviceID, engine);
        }
    }

    class ComputeGraphTest2<T> : TestEx<T>, IComputeGraphTest2
    {
        Random m_random = new Random(3);
        Blob<T> m_A;
        Blob<T> m_B;
        Blob<T> m_C;

        public ComputeGraphTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
            m_A = new Blob<T>(m_cuda, m_log);
            m_A.Name = "A";
            m_B = new Blob<T>(m_cuda, m_log);
            m_B.Name = "B";
            m_C = new Blob<T>(m_cuda, m_log);
            m_C.Name = "C";
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            m_A.Dispose();
            m_B.Dispose();
            m_C.Dispose();
            base.dispose();
        }

        private Tuple<Blob<T>, WeightMatrix> createMatrix(string strName, Blob<T> b, int nM, int nN, int nAxis = 0)
        {
            b.Reshape(nM, nN, 1, 1);
            b.Name = strName;
            WeightMatrix mtx = new WeightMatrix(nM, nN, 0, strName);

            double[] rgData = convert(b.mutable_cpu_data);

            for (int i = 0; i < rgData.Length; i++)
            {
                double dfVal = m_random.NextDouble();
                rgData[i] = dfVal;
                mtx.Weight[i] = dfVal;
            }

            b.mutable_cpu_data = convert(rgData);

            return new Tuple<Blob<T>, WeightMatrix>(b, mtx);
        }

        private void fillGradient(Blob<T> b, WeightMatrix mtx)
        {
            Random random = new Random(3);

            double[] rgDiff = null;
            
            if (b != null)
                rgDiff = convert(b.mutable_cpu_diff);

            for (int i = 0; i < mtx.Gradient.Length; i++)
            {
                double dfVal = m_random.NextDouble();
                if (rgDiff != null)
                    rgDiff[i] = dfVal;
                mtx.Gradient[i] = dfVal;
            }

            if (b != null)
                b.mutable_cpu_diff = convert(rgDiff);
        }

        private void fillGradient(Blob<T> b, WeightMatrix mtx, double[] rgVal)
        {
            double[] rgData = null;

            if (b != null)
                rgData = convert(b.mutable_cpu_diff);

            for (int i = 0; i < mtx.Gradient.Length; i++)
            {
                double dfVal = rgVal[i];
                if (rgData != null)
                    rgData[i] = dfVal;
                mtx.Gradient[i] = dfVal;
            }

            if (b != null)
                b.mutable_cpu_diff = convert(rgData);
        }

        private void fillGradient(Blob<T> b, WeightMatrix mtx, double dfVal)
        {
            double[] rgData = null;

            if (b != null)
                rgData = convert(b.mutable_cpu_diff);

            for (int i = 0; i < mtx.Gradient.Length; i++)
            {
                if (rgData != null)
                    rgData[i] = dfVal;
                mtx.Gradient[i] = dfVal;
            }

            if (b != null)
                b.mutable_cpu_diff = convert(rgData);
        }

        private void fillGradient(Tuple<Blob<T>, WeightMatrix> val, double[] rgVal)
        {
            fillGradient(val.Item1, val.Item2, rgVal);
        }

        private void fillGradient(Tuple<Blob<T>, WeightMatrix> val, double dfVal)
        {
            fillGradient(val.Item1, val.Item2, dfVal);
        }

        private void fillWeight(Blob<T> b, WeightMatrix mtx, double[] rgVal)
        {
            double[] rgData = null;

            if (b != null)
                rgData = convert(b.mutable_cpu_data);

            for (int i = 0; i < mtx.Weight.Length; i++)
            {
                double dfVal = rgVal[i];
                if (rgData != null)
                    rgData[i] = dfVal;
                mtx.Weight[i] = dfVal;
            }

            if (b != null)
                b.mutable_cpu_data = convert(rgData);
        }

        private void fillWeight(Blob<T> b, WeightMatrix mtx, double dfVal)
        {
            double[] rgData = null;

            if (b != null)
                rgData = convert(b.mutable_cpu_data);

            for (int i = 0; i < mtx.Weight.Length; i++)
            {
                if (rgData != null)
                    rgData[i] = dfVal;
                mtx.Weight[i] = dfVal;
            }

            if (b != null)
                b.mutable_cpu_data = convert(rgData);
        }

        private void fillWeight(Tuple<Blob<T>, WeightMatrix> val, double[] rgVal)
        {
            fillWeight(val.Item1, val.Item2, rgVal);
        }

        private void fillWeight(Tuple<Blob<T>, WeightMatrix> val, double dfVal)
        {
            fillWeight(val.Item1, val.Item2, dfVal);
        }

        private bool compare(Blob<T> b, WeightMatrix mtx, bool bDiff = false)
        {
            double[] rgBlob = (bDiff) ? convert(b.mutable_cpu_diff) : convert(b.mutable_cpu_data);
            double[] rgMtx = (bDiff) ? mtx.Gradient : mtx.Weight;

            m_log.CHECK_EQ(rgBlob.Length, rgMtx.Length, "The blob and matrix should have the same number of items!");

            for (int i = 0; i < rgBlob.Length; i++)
            {
                double dfBlob = rgBlob[i];
                double dfMtx = rgMtx[i];
                m_log.EXPECT_NEAR(dfBlob, dfMtx, 0.000001, "The values at index " + i.ToString() + " are not equal!");
            }

            return true;
        }

        public void TestPeekRow()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nRow = 3;
            int nCol = 2;
            Tuple<Blob<T>, WeightMatrix> val = createMatrix("val", m_A, nRow, nCol, nAxis);

            for (int i = 0; i < nRow; i++)
            {
                g.PeekRow(val.Item1, m_B, i);
                WeightMatrix res = g1.PeekRow(val.Item2, i);
                compare(m_B, res);

                fillGradient(m_B, res);
                g.Backward(true);
                g1.backward(true);
            }

            compare(m_A, val.Item2, true);
        }

        public void Test_tanh()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nRow = 3;
            int nCol = 2;
            Tuple<Blob<T>, WeightMatrix> val = createMatrix("val", m_A, nRow, nCol, nAxis);

            g.tanh(val.Item1, m_B);
            WeightMatrix res = g1.tanh(val.Item2);

            compare(m_B, res);

            fillGradient(m_B, res);

            g.Backward();
            g1.backward();

            compare(m_A, val.Item2, true);
        }

        public void Test_sigmoid()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nRow = 3;
            int nCol = 2;
            Tuple<Blob<T>, WeightMatrix> val = createMatrix("val", m_A, nRow, nCol, nAxis);

            g.sigmoid(val.Item1, m_B);
            WeightMatrix res = g1.sigmoid(val.Item2);

            compare(m_B, res);

            fillGradient(m_B, res);

            g.Backward();
            g1.backward();

            compare(m_A, val.Item2, true);
        }

        public void Test_eltmul()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nRow = 2;
            int nCol = 3;
            Tuple<Blob<T>, WeightMatrix> valA = createMatrix("val", m_A, nRow, nCol, nAxis);
            Tuple<Blob<T>, WeightMatrix> valB = createMatrix("val", m_B, nRow, nCol, nAxis);

            g.eltmul(m_A, m_B, m_C);
            WeightMatrix res = g1.eltmul(valA.Item2, valB.Item2);

            compare(m_C, res);

            fillGradient(m_C, res);

            g.Backward();
            g1.backward();

            compare(m_A, valA.Item2, true);
            compare(m_B, valB.Item2, true);
        }

        public void Test_scalemul()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nRow = 2;
            int nCol = 3;
            Tuple<Blob<T>, WeightMatrix> valA = createMatrix("val", m_A, nRow, nCol, nAxis);
            Tuple<Blob<T>, WeightMatrix> valB = createMatrix("val", m_B, nRow, nCol, nAxis);

            g.scalemul(m_A, m_B, m_C);
            WeightMatrix res = g1.scalemul(valA.Item2, valB.Item2);

            compare(m_C, res);

            fillGradient(m_C, res);

            g.Backward();
            g1.backward();

            compare(m_A, valA.Item2, true);
            compare(m_B, valB.Item2, true);
        }

        public void Test_mul()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nM = 2;
            int nN = 3;
            int nK = 4;
            Tuple<Blob<T>, WeightMatrix> valA = createMatrix("val", m_A, nM, nK, nAxis);
            Tuple<Blob<T>, WeightMatrix> valB = createMatrix("val", m_B, nK, nN, nAxis);

            g.mul(m_A, m_B, m_C);
            WeightMatrix res = g1.mul(valA.Item2, valB.Item2);

            m_log.CHECK_EQ(m_C.num, nM, "The C.num should equal M=" + nM.ToString());
            m_log.CHECK_EQ(m_C.channels, nN, "The C.channels should equal N=" + nN.ToString());

            compare(m_C, res);

            fillGradient(m_C, res);

            g.Backward();
            g1.backward();

            compare(m_A, valA.Item2, true);
            compare(m_B, valB.Item2, true);
        }

        public void Test_mul2()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nM = 1;
            int nN = 16;
            int nK = 8;
            Tuple<Blob<T>, WeightMatrix> valA = createMatrix("val", m_A, nM, nK, nAxis);
            Tuple<Blob<T>, WeightMatrix> valB = createMatrix("val", m_B, nK, nN, nAxis);

            fillWeight(valA, new double[] { -0.875, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125 });
            fillWeight(valB, 0.1);

            g.mul(m_A, m_B, m_C);
            WeightMatrix res = g1.mul(valA.Item2, valB.Item2);

            m_log.CHECK_EQ(m_C.num, nM, "The C.num should equal M=" + nM.ToString());
            m_log.CHECK_EQ(m_C.channels, nN, "The C.channels should equal N=" + nN.ToString());

            compare(m_C, res);

            fillGradient(new Tuple<Blob<T>, WeightMatrix>(m_C, res), new double[] { 0.250, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 1, 1, 1, 1, 1, 1, 1, 1 });

            g.Backward();
            g1.backward();

            compare(m_A, valA.Item2, true);
            compare(m_B, valB.Item2, true);
        }

        public void Test_add()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nRow = 2;
            int nCol = 3;
            Tuple<Blob<T>, WeightMatrix> valA = createMatrix("val", m_A, nRow, nCol, nAxis);
            Tuple<Blob<T>, WeightMatrix> valB = createMatrix("val", m_B, nRow, nCol, nAxis);

            g.add(m_A, m_B, m_C);
            WeightMatrix res = g1.add(valA.Item2, valB.Item2);

            compare(m_C, res);

            fillGradient(m_C, res);

            g.Backward();
            g1.backward();

            compare(m_A, valA.Item2, true);
            compare(m_B, valB.Item2, true);
        }

        public void Test_softmax()
        {
            int nAxis = 0;
            ComputeGraph<T> g = new ComputeGraph<T>(m_cuda, m_log, nAxis);
            ComputeGraphCpu g1 = new ComputeGraphCpu();

            int nRow = 3;
            int nCol = 1;
            Tuple<Blob<T>, WeightMatrix> val = createMatrix("val", m_A, nRow, nCol, nAxis);

            g.softmax(val.Item1, m_B);
            WeightMatrix res = g1.Softmax(val.Item2);

            compare(m_B, res);

            fillGradient(m_B, res);

            g.Backward();
            g1.backward();

            compare(m_A, val.Item2, true);
        }
    }

    /// <summary>
    /// RandomGenerator used with CPU tests.
    /// </summary>
    /// <remarks>
    /// Distributed by GitHub:mashmawy/Seq2SeqLearn under MIT License at
    /// @see [mashmawy/Seq2SeqLearn](https://github.com/mashmawy/Seq2SeqLearn)
    /// </remarks>
    static class RandomGenerator
    {

        public static bool Return_V { get; set; }
        public static double V_Val { get; set; }

        private static Random random = new Random(3);

        public static double GaussRandom()
        {
            if (Return_V)
            {
                Return_V = false;
                return V_Val;
            }
            var u = 2 * random.NextDouble() - 1;
            var v = 2 * random.NextDouble() - 1;
            var r = (u * u) + (v * v);

            if (r == 0 || r > 1) return GaussRandom();
            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            V_Val = v * c;
            Return_V = true;
            return u * c;
        }

        public static double NormalRandom(double mu, double std)
        {
            return mu + GaussRandom() * std;
        }
    }

    /// <summary>
    /// WeightMatrix used with CPU tests.
    /// </summary>
    /// <remarks>
    /// Distributed by GitHub:mashmawy/Seq2SeqLearn under MIT License at
    /// @see [mashmawy/Seq2SeqLearn](https://github.com/mashmawy/Seq2SeqLearn)
    /// </remarks>
    class WeightMatrix
    {
        public string Name { get; set; }
        public int Rows { get; set; }
        public int Columns { get; set; }
        public double[] Weight { get; set; }
        public double[] Gradient { get; set; }
        public double[] Cash { get; set; }

        public WeightMatrix()
        {

        }
        public WeightMatrix(double[] weights, string strName = "")
        {
            this.Name = strName;
            this.Rows = weights.Length;
            this.Columns = 1;
            this.Weight = new double[this.Rows];
            this.Gradient = new double[this.Rows];
            this.Cash = new double[this.Rows];
            this.Weight = weights;

        }

        public WeightMatrix(int rows, int columns, string strName = "", bool normal = false)
        {
            this.Name = strName;
            this.Rows = rows;
            this.Columns = columns;
            var n = rows * columns;
            this.Weight = new double[n];
            this.Gradient = new double[n];
            this.Cash = new double[n];

            var scale = Math.Sqrt(1.0 / (rows * columns));
            if (normal)
            {
                scale = 0.08;
            }
            for (int i = 0; i < n; i++)
            {
                this.Weight[i] = RandomGenerator.NormalRandom(0.0, scale);
            }
        }

        public WeightMatrix(int rows, int columns, double c, string strName = "")
        {
            this.Name = strName;
            this.Rows = rows;
            this.Columns = columns;
            var n = rows * columns;
            this.Weight = new double[n];
            this.Gradient = new double[n];

            this.Cash = new double[n];
            for (int i = 0; i < n; i++)
            {
                this.Weight[i] = c;
            }
        }

        public override string ToString()
        {

            return "{" + Rows.ToString() + "," + Columns.ToString() + "}";
        }
        public double Get(int x, int y)
        {
            var ix = ((this.Columns * x) + y);
            return this.Weight[ix];
        }

        public void Set(int x, int y, double v)
        {
            var ix = ((this.Columns * x) + y);
            this.Weight[ix] = v;
        }

        public void Add(int x, int y, double v)
        {
            var ix = ((this.Columns * x) + y);
            this.Weight[ix] += v;
        }

        public double Get_Grad(int x, int y)
        {
            var ix = ((this.Columns * x) + y);
            return this.Gradient[ix];
        }

        public void Set_Grad(int x, int y, double v)
        {
            var ix = ((this.Columns * x) + y);
            this.Gradient[ix] = v;
        }

        public void Add_Grad(int x, int y, double v)
        {
            var ix = ((this.Columns * x) + y);
            this.Gradient[ix] += v;
        }

        public WeightMatrix CloneAndZero()
        {
            return new WeightMatrix(this.Rows, this.Columns, 0.0);

        }

        public WeightMatrix Clone()
        {
            var v = new WeightMatrix(this.Rows, this.Columns, 0.0);
            var n = this.Weight.Length;
            for (int i = 0; i < n; i++)
            {
                v.Weight[i] = this.Weight[i];
            }
            return v;
        }
    }

    /// <summary>
    /// ComputeGraph used with CPU tests.
    /// </summary>
    /// <remarks>
    /// Distributed by GitHub:mashmawy/Seq2SeqLearn under MIT License at
    /// @see [mashmawy/Seq2SeqLearn](https://github.com/mashmawy/Seq2SeqLearn)
    /// </remarks>
    class ComputeGraphCpu
    {
        List<Action> backprop = new List<Action>();

        public bool needs_backprop { get; set; }
        public ComputeGraphCpu(bool needBack = true)
        {
            this.needs_backprop = needBack;
        }
        public WeightMatrix tanh(WeightMatrix m)
        {
            // tanh nonlinearity
            var res = new WeightMatrix(m.Rows, m.Columns, 0);
            var n = m.Weight.Length;
            for (var i = 0; i < n; i++)
            {
                res.Weight[i] = Math.Tanh(m.Weight[i]);
            }

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < n; i++)
                    {
                        // grad for z = tanh(x) is (1 - z^2)
                        var mwi = res.Weight[i];
                        m.Gradient[i] += (1.0 - mwi * mwi) * res.Gradient[i];
                    }
                };
                this.backprop.Add(backward);
            }
            return res;
        }
        public WeightMatrix PeekRow(WeightMatrix m, int ix)
        {
            var d = m.Columns;
            var res = new WeightMatrix(1, d, 0);
            for (int i = 0, n = d; i < n; i++) { res.Weight[i] = m.Weight[d * ix + i]; } // copy over the data

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    for (int i = 0, n = d; i < n; i++) { m.Gradient[d * ix + i] += res.Gradient[i]; }
                };
                this.backprop.Add(backward);
            }
            return res;
        }
        private double sig(double x)
        {
            // helper function for computing sigmoid
            return 1.0 / (1 + Math.Exp(-x));
        }
        public WeightMatrix sigmoid(WeightMatrix m)
        {
            // sigmoid nonlinearity
            WeightMatrix res = new WeightMatrix(m.Rows, m.Columns, 0);
            var n = m.Weight.Length;
            for (var i = 0; i < n; i++)
            {
                res.Weight[i] = sig(m.Weight[i]);
            }

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < n; i++)
                    {
                        // grad for z = tanh(x) is (1 - z^2)
                        var mwi = res.Weight[i];
                        m.Gradient[i] += mwi * (1.0 - mwi) * res.Gradient[i];
                    }
                };
                this.backprop.Add(backward);
            }
            return res;
        }
        public WeightMatrix mul(WeightMatrix m1, WeightMatrix m2)
        {
            var n = m1.Rows;
            var d = m2.Columns;
            var res = new WeightMatrix(n, d, 0);
            for (var i = 0; i < m1.Rows; i++)
            { // loop over rows of m1
                for (var j = 0; j < m2.Columns; j++)
                { // loop over cols of m2
                    var dot = 0.0;
                    for (var k = 0; k < m1.Columns; k++)
                    { // dot product loop
                        dot += m1.Weight[m1.Columns * i + k] * m2.Weight[m2.Columns * k + j];
                    }
                    res.Weight[d * i + j] = dot;
                }
            }
            //  var res = mulParalel(m1, m2);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < m1.Rows; i++)
                    { // loop over rows of m1
                        for (var j = 0; j < m2.Columns; j++)
                        { // loop over cols of m2
                            for (var k = 0; k < m1.Columns; k++)
                            { // dot product loop
                                int nIdxRes = d * i + j;
                                var b = res.Gradient[d * i + j];

                                int nWt2Idx = m2.Columns * k + j;
                                var m2wt = m2.Weight[m2.Columns * k + j];
                                var diff1 = m2wt * b;

                                int nM1GradIdx = m1.Columns * i + k;
                                m1.Gradient[m1.Columns * i + k] += diff1;

                                var m1wt = m1.Weight[m1.Columns * i + k];
                                var diff2 = m1wt * b;
                                m2.Gradient[m2.Columns * k + j] += diff2;
                            }
                        }
                    }
                };
                this.backprop.Add(backward);
            }
            return res;
        }
        public WeightMatrix add(WeightMatrix m1, WeightMatrix m2)
        {

            var res = new WeightMatrix(m1.Rows, m1.Columns, 0);
            for (int i = 0, n = m1.Weight.Length; i < n; i++)
            {
                res.Weight[i] = m1.Weight[i] + m2.Weight[i];
            }
            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    for (int i = 0, n = m1.Weight.Length; i < n; i++)
                    {
                        m1.Gradient[i] += res.Gradient[i];
                        m2.Gradient[i] += res.Gradient[i];
                    }
                };
                this.backprop.Add(backward);
            }
            return res;

        }
        public WeightMatrix eltmul(WeightMatrix m1, WeightMatrix m2)
        {

            var res = new WeightMatrix(m1.Rows, m1.Columns, 0);
            for (int i = 0, n = m1.Weight.Length; i < n; i++)
            {
                res.Weight[i] = m1.Weight[i] * m2.Weight[i];
            }
            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    for (int i = 0, n = m1.Weight.Length; i < n; i++)
                    {
                        m1.Gradient[i] += m2.Weight[i] * res.Gradient[i];
                        m2.Gradient[i] += m1.Weight[i] * res.Gradient[i];
                    }
                };
                this.backprop.Add(backward);
            }
            return res;
        }
        public WeightMatrix scalemul(WeightMatrix m1, WeightMatrix m2)
        {

            var res = new WeightMatrix(m1.Rows, m1.Columns, 0);
            for (int i = 0, n = m1.Weight.Length; i < n; i++)
            {
                res.Weight[i] = m1.Weight[i] * m2.Weight[0];
            }
            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    for (int i = 0, n = m1.Weight.Length; i < n; i++)
                    {
                        m1.Gradient[i] += m2.Weight[0] * res.Gradient[i];
                        m2.Gradient[0] += m1.Weight[i] * res.Gradient[i];

                    }
                };
                this.backprop.Add(backward);
            }
            return res;
        }
        public WeightMatrix Softmax(WeightMatrix m)
        {
            var res = new WeightMatrix(m.Rows, m.Columns, 0); // probability volume
            var maxval = -999999.0;
            for (int i = 0, n = m.Weight.Length; i < n; i++)
            {
                if (m.Weight[i] > maxval) maxval = m.Weight[i];
            }

            var s = 0.0;
            for (int i = 0, n = m.Weight.Length; i < n; i++)
            {
                res.Weight[i] = Math.Exp(m.Weight[i] - maxval);
                s += res.Weight[i];
            }
            for (int i = 0, n = m.Weight.Length; i < n; i++) { res.Weight[i] /= s; }



            if (this.needs_backprop)
            {
                Action backward = () =>
                {

                    double ss = 0.0;
                    for (int ix = 0; ix < m.Weight.Length; ix++)
                    {
                        m.Gradient[ix] = res.Gradient[ix] * res.Weight[ix];
                        ss += res.Gradient[ix] * res.Weight[ix];
                    }
                    for (int ix = 0; ix < m.Weight.Length; ix++)
                    {
                        m.Gradient[ix] -= ss * res.Weight[ix];
                    }


                    //for (int i = 0; i < m.Count; i++)
                    //{
                    //    m[i].Gradient[0]  = res[i].Gradient[0] * res[i].Weight[0];

                    //    ss += res[i].Gradient[0] * res[i].Weight[0];
                    //}
                    //for (int i = 0; i < m.Count; i++)
                    //{
                    //    m[i].Gradient[0] -= ss * res[i].Weight[0];

                    //}
                };
                this.backprop.Add(backward);
            }
            return res;
        }
        public void backward(bool bClear = false)
        {
            for (var i = this.backprop.Count - 1; i >= 0; i--)
            {
                string strInfo = this.backprop[i].Method.ToString();
                Trace.WriteLine(strInfo);

                this.backprop[i](); // tick!
            }

            if (bClear) 
                this.backprop.Clear();
        }
    }
}
