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
        DEFAULT,
        VALUES,
        BLOB
    }

    public interface IXMyCaffeGym
    {
        void Initialize(Log log, PropertySet properties);
        void Close();
        IXMyCaffeGym Clone(PropertySet properties = null);
        string Name { get; }
        Tuple<State, double, bool> Reset();
        Tuple<State, double, bool> Step(int nAction);
        Bitmap Render(int nWidth, int nHeight, out Bitmap bmpAction);
        Bitmap Render(int nWidth, int nHeight, double[] rgData, out Bitmap bmpAction);
        Dictionary<string, int> GetActionSpace();
        DatasetDescriptor GetDataset(DATA_TYPE dt);
        int UiDelay { get; }
        DATA_TYPE SelectedDataType { get; }

        DATA_TYPE[] SupportedDataType { get; }
    }

    public abstract class State
    {
        public State()
        {
        }

        public abstract State Clone();
        public abstract Tuple<double, double, double, bool>[] ToArray();
    }
}
