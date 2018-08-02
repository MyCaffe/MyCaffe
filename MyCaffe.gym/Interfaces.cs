using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public interface IxMycaffeGym
    {
        void Initialize(Log log);
        void Close();
        string Name { get; }
        void Reset();
        Tuple<double[], double, bool> Step();
        Bitmap Render(int nWidth, int nHeight);
        void AddAction(int nAction);
        Dictionary<string, int> GetActionSpace();
    }

    public class State
    {
        public State()
        {
        }
    }
}
