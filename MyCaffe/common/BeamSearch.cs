using MyCaffe.basecode;
using MyCaffe.layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The BeamSearch uses the softmax output from the network and continually runs the
    /// net on each output (using the output as input) until the end of token is reached.
    /// The beam-search tree is returned.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class BeamSearch<T>
    {
        Net<T> m_net;
        Layer<T> m_layer = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="net">Specifies the net used for the forward passes.</param>
        public BeamSearch(Net<T> net)
        {
            m_net = net;

            foreach (Layer<T> layer1 in m_net.layers)
            {
                if (layer1.SupportsPreProcessing && layer1.SupportsPostProcessing)
                {
                    m_layer = layer1;
                    break;
                }
            }

            if (m_layer == null)
                throw new Exception("At least one layer in the net must support pre and post processing!");
        }

        /// <summary>
        /// Perform the beam-search.
        /// </summary>
        /// <param name="input">Specifies the input data (e.g. the encoder input)</param>
        /// <param name="nK">Specifies the beam width for the search.</param>
        /// <param name="dfThreshold">Specifies the threshold where detected items with probabilities less than the threshold are ignored (default = 0.2).</param>
        /// <param name="nMax">Specifies the maximum length to process (default = 80)</param>
        /// <returns>The list of top sequences is returned.</returns>
        /// <remarks>
        /// The beam-search algorithm is inspired by the article
        /// @see [How to Implement a Beam Search Decoder for Natural Language Processing](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/) by Jason Brownlee, "Machine Learning Mastery", 2018
        /// </remarks>
        public List<Tuple<double, List<Tuple<string, int, double>>>> Search(PropertySet input, int nK, double dfThreshold = 0.2, int nMax = 80)
        {
            List<Tuple<double, List<Tuple<string, int, double>>>> rgSequences = new List<Tuple<double, List<Tuple<string, int, double>>>>();
            rgSequences.Add(new Tuple<double, List<Tuple<string, int, double>>>(0, new List<Tuple<string, int, double>>()));

            List<Tuple<double, List<Tuple<string, int, double>>>> rgAllCandidates = new List<Tuple<double, List<Tuple<string, int, double>>>>();

            BlobCollection<T> colBottom = m_layer.PreProcessInput(input, null);
            double dfLoss;
            string strInput = input.GetProperty("InputData");
            bool bDone = false;

            BlobCollection<T> colTop = m_net.Forward(colBottom, out dfLoss);
            List<Tuple<string, int, double>> rgRes = m_layer.PostProcessOutput(colTop[0], nK);
            rgRes = rgRes.Where(p => p.Item3 >= dfThreshold).ToList();
            List<List<Tuple<string, int, double>>> rgrgRes = new List<List<Tuple<string, int, double>>>();

            rgrgRes.Add(rgRes);

            while (!bDone && nMax > 0)
            {
                int nProcessedCount = 0;

                List<Tuple<double, List<Tuple<string, int, double>>>> rgCandidates = new List<Tuple<double, List<Tuple<string, int, double>>>>();

                for (int i = 0; i < rgSequences.Count; i++)
                {
                    if (rgrgRes[i].Count > 0)
                    {
                        for (int j = 0; j < rgrgRes[i].Count; j++)
                        {
                            if (rgrgRes[i][j].Item1.Length > 0)
                            {
                                double dfScore = rgSequences[i].Item1 - Math.Log(rgrgRes[i][j].Item3);

                                List<Tuple<string, int, double>> rgSequence1 = new List<Tuple<string, int, double>>();
                                rgSequence1.AddRange(rgSequences[i].Item2);
                                rgSequence1.Add(rgrgRes[i][j]);

                                rgCandidates.Add(new Tuple<double, List<Tuple<string, int, double>>>(dfScore, rgSequence1));
                                nProcessedCount++;
                            }
                        }
                    }
                    else
                    {
                        rgCandidates.Add(rgSequences[i]);
                    }
                }

                if (nProcessedCount > 0)
                {
                    rgSequences = rgCandidates.OrderBy(p => p.Item1).Take(nK).ToList();
                    rgrgRes = new List<List<Tuple<string, int, double>>>();
                    Dictionary<int, List<Tuple<string, int, double>>> rgProcessed = new Dictionary<int, List<Tuple<string, int, double>>>();

                    for (int i = 0; i < rgSequences.Count; i++)
                    {
                        rgRes = new List<Tuple<string, int, double>>();
                        int nIdx = rgSequences[i].Item2.Last().Item2;

                        if (!rgProcessed.ContainsKey(nIdx))
                        {
                            m_layer.PreProcessInput(strInput, nIdx, colBottom);
                            colTop = m_net.Forward(colBottom, out dfLoss, true);
                            List<Tuple<string, int, double>> rgRes1 = m_layer.PostProcessOutput(colTop[0], nK);
                            rgRes1 = rgRes1.Where(p => p.Item3 >= dfThreshold).ToList();

                            for (int j = 0; j < rgRes1.Count; j++)
                            {
                                if (rgRes1[j].Item1.Length > 0)
                                    rgRes.Add(rgRes1[j]);
                                else
                                    Trace.WriteLine("EOS");
                            }

                            rgrgRes.Add(rgRes);
                        }
                        else
                        {
                            rgrgRes.Add(rgProcessed[nIdx]);
                        }
                    }
                }
                else
                {
                    bDone = true;
                }

                nMax--;
            }

            return rgSequences;
        }
    }
}
