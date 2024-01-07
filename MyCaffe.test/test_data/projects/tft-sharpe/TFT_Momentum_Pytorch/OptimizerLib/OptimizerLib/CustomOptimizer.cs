using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Xml.Linq;

namespace OptimizerLib
{
    public class CustomOptimizer
    {
        float m_fLr;
        float m_fDecay;
        float m_fBeta1;
        float m_fBeta2;
        int m_nStep;
        float m_fBiasCorrection1;
        float m_fBiasCorrection2;
        float m_fStepSize;
        float m_fEps;
        float[] m_rgM;
        float[] m_rgV;
        Random m_random = new Random(1701);

        public CustomOptimizer(int nPid)
        {
            Console.WriteLine("Debug Python process id: {0}", nPid);
        }

        public float[] m
        {
            get { return m_rgM; }
        }

        public float[] v
        {
            get { return m_rgV; }
        }

        public int get_next_index(int nMax)
        {
            return m_random.Next(nMax);
        }

        public void update_step(float fLr, float fDecay, float fBeta1, float fBeta2, int nT, float fEps)
        {
            m_fLr = fLr;
            m_fDecay = fDecay;
            m_fBeta1 = fBeta1;
            m_fBeta2 = fBeta2;
            m_fEps = fEps;
            m_nStep = nT;

            m_fBiasCorrection1 = 1 - (float)Math.Pow(m_fBeta1, m_nStep);
            m_fBiasCorrection2 = 1 - (float)Math.Pow(m_fBeta2, m_nStep);
            m_fStepSize = m_fLr / m_fBiasCorrection1;
        }

        public float[] step(float[] rgW, float[] rgG, float[] rgM, float[] rgV)
        {
            float fSum = 0;
            for (int i = 0; i < rgW.Length; i++)
            {
                rgW[i] *= (1 - m_fLr * m_fDecay);
                rgM[i] = rgM[i] * m_fBeta1 + rgG[i] * (1 - m_fBeta1);
                rgV[i] = rgV[i] * m_fBeta2 + rgG[i] * rgG[i] * (1 - m_fBeta2);

                float fDenom = (float)Math.Sqrt(rgV[i]) / (float)Math.Sqrt(m_fBiasCorrection2) + m_fEps;
                rgW[i] = rgW[i] + (-1 * m_fStepSize) * rgM[i] / fDenom;
                fSum += Math.Abs(rgW[i]);
            }

            m_rgM = rgM;
            m_rgV = rgV;

            return rgW;
        }
    }
}
