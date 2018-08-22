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
        void Initialize(Log log);
        void Close();
        string Name { get; }
        void Reset();
        Tuple<Tuple<double,double,double>[], double, bool> Step();
        Bitmap Render(int nWidth, int nHeight);
        void AddAction(int nAction);
        Dictionary<string, int> GetActionSpace();
        DatasetDescriptor GetDataset(DATA_TYPE dt);
    }

    public class State
    {
        public State()
        {
        }
    }
}
