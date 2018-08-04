using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public class GymCollection : IEnumerable<IXMyCaffeGym>
    {
        List<IXMyCaffeGym> m_rgGym = new List<IXMyCaffeGym>();

        public GymCollection()
        {
        }

        public void Load()
        {
            m_rgGym.Add(new CartPoleGym());
        }

        public IXMyCaffeGym Find(string strName)
        {
            foreach (IXMyCaffeGym igym in m_rgGym)
            {
                if (igym.Name == strName)
                    return igym;
            }

            return null;
        }

        public IEnumerator<IXMyCaffeGym> GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }
    }
}
