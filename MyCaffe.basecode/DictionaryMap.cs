using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyCaffe.basecode
{
#pragma warning disable 1591

    public class DictionaryMap<T> /** @private */
    {
        Dictionary<int, T> m_rgMap = new Dictionary<int, T>();
        T m_dfDefault;

        public DictionaryMap(T dfDefault)
        {
            m_dfDefault = dfDefault;
        }

        public DictionaryMap(int nCount, T dfDefault)
        {
            m_dfDefault = dfDefault;

            for (int i = 0; i < nCount; i++)
            {
                m_rgMap.Add(i, dfDefault);
            }
        }

        public T this[int nIdx]
        {
            get
            {
                if (m_rgMap.ContainsKey(nIdx))
                    return m_rgMap[nIdx];

                return m_dfDefault;
            }

            set
            {
                if (m_rgMap.ContainsKey(nIdx))
                    m_rgMap[nIdx] = value;
                else
                    m_rgMap.Add(nIdx, value);
            }
        }

        public int Count
        {
            get { return m_rgMap.Count; }
        }

        public void Clear()
        {
            m_rgMap.Clear();
        }

        public Dictionary<int, T> Map
        {
            get { return m_rgMap; }
        }
    }
#pragma warning restore 1591
}
