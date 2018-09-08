using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public enum DATA_TYPE
    {
        VALUES,
        BLOB
    }

    public interface IXMyCaffeGym
    {
        void Initialize(Log log, double[] rgdfInit);
        void Close();
        IXMyCaffeGym Clone();
        string Name { get; }
        Tuple<State, double, bool> Reset();
        Tuple<State, double, bool> Step(int nAction);
        Bitmap Render(int nWidth, int nHeight, out Bitmap bmpAction);
        Dictionary<string, int> GetActionSpace();
        DatasetDescriptor GetDataset(DATA_TYPE dt);
    }

    public abstract class State
    {
        public State()
        {
        }

        public abstract State Clone();
        public abstract Tuple<double, double, double>[] ToArray();
    }
}
