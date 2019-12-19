using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyCaffe.common
{
#pragma warning disable 1591

    public class DictionaryEx<TKey, TValue> : Dictionary<TKey, TValue> /** @private */
    {
        TValue m_tDefault;

        public DictionaryEx(TValue tDefault)
            : base()
        {
            m_tDefault = tDefault;
        }

        public new TValue this[TKey k]
        {
            get 
            {
                if (!Keys.Contains(k))
                    Add(k, m_tDefault);

                return base[k]; 
            }
            set
            {
                if (!Keys.Contains(k))
                    Add(k, value);
                else
                    base[k] = value;
            }
        }
    }

#pragma warning restore 1591
}
