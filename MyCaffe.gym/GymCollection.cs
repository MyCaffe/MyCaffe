using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public class GymCollection : IEnumerable<IxMycaffeGym>
    {
        List<IxMycaffeGym> m_rgGym = new List<IxMycaffeGym>();

        public GymCollection()
        {
        }

        public void Load()
        {
            m_rgGym.Add(new CartPoleGym());
        }

        public IxMycaffeGym Find(string strName)
        {
            foreach (IxMycaffeGym igym in m_rgGym)
            {
                if (igym.Name == strName)
                    return igym;
            }

            return null;
        }

        public IEnumerator<IxMycaffeGym> GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }
    }
}
