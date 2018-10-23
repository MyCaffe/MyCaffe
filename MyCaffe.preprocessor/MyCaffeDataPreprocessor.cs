using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.stream;

namespace MyCaffe.preprocessor
{
    public partial class MyCaffeDataPreprocessor<T> : Component, IXPreprocessor<T>
    {
        public MyCaffeDataPreprocessor()
        {
            InitializeComponent();
        }

        public MyCaffeDataPreprocessor(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        public void Initialize(IXMyCaffe<T> imycaffe, IXStreamDatabase istrm, string strPreProcessorDLLPath)
        {
            throw new NotImplementedException();
        }

        public Tuple<Bitmap, SimpleDatum> Render(SimpleDatum sd)
        {
            throw new NotImplementedException();
        }

        public void Reset(int nStartOffset = 0)
        {
            throw new NotImplementedException();
        }

        public void Shutdown()
        {
            throw new NotImplementedException();
        }

        public Tuple<Blob<T>, SimpleDatum> Step(bool bGetSimpleDatum, int nWait = 1000)
        {
            throw new NotImplementedException();
        }
    }
}
