using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace MyCaffe.layers.gpt.layers.gpt
{
    /// <summary>
    /// The VocabularyBytePairEncoding class manages the data vocabulary of byte-pair encoded tokens for LLaMA2 based models.
    /// </summary>
    /// <remarks>
    /// @see [GitHub belladoreai/llama-tokenizer.js](https://github.com/belladoreai/llama-tokenizer-js/tree/master) by belladore.ai, 2023, GitHub.
    /// Distributed under the MIT license at https://github.com/belladoreai/llama-tokenizer-js/blob/master/LICENSE.md
    /// </remarks>
    public class VocabularyBytePairEncoding : IVocabulary
    {
        Dictionary<string, int> m_rgVocabByString = new Dictionary<string, int>();
        Dictionary<int, string> m_rgVocabById = new Dictionary<int, string>();
        Dictionary<string, int> m_rgMerges = new Dictionary<string, int>();
        bool m_bAddBos = true;
        bool m_bAddPrecedingSpace = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        public VocabularyBytePairEncoding()
        {
            m_rgVocabById = decodeVocabulary(Properties.Resources.llama_vocab_base64);
            foreach (KeyValuePair<int, string> kvp in m_rgVocabById)
            {
                m_rgVocabByString[kvp.Value] = kvp.Key;
            }
            m_rgMerges = decodeMerges(Properties.Resources.llama_merges_binary);
        }

        private string getMergeIdentifyingString(int nId1, int nId2)
        {
            return m_rgVocabById[nId1] + " " + m_rgVocabById[nId2];
        }

        private Dictionary<string, int> decodeMerges(string strMerges)
        {
            byte[] rgData = Convert.FromBase64String(strMerges);

            // Each byte-pair represents a tokenId.
            // Convert byte-pairs to tokenIds (integers between 0 and 32000)
            int[] rgTokenId = new int[rgData.Length / 2];
            for (int i = 0; i < rgData.Length; i += 2)
            {
                int nTokenId = BitConverter.ToInt16(rgData, i);
                rgTokenId[i/2] = nTokenId;
            }

            // Each pair of tokenIds represents a merge.
            Dictionary<string, int> rgMerges = new Dictionary<string, int>();
            for (int i = 0; i < rgTokenId.Length; i += 2)
            {
                int nId1 = rgTokenId[i];
                int nId2 = rgTokenId[i + 1];
                string strMerge = getMergeIdentifyingString(nId1, nId2);
                rgMerges[strMerge] = i+1;
            }

            return rgMerges;
        }

        private Dictionary<int, string> decodeVocabulary(string strVocab)
        {
            byte[] rgData = Convert.FromBase64String(strVocab);
            string str = Encoding.UTF8.GetString(rgData);
            string[] rgstr = str.Split(new char[] { '\n' });

            Dictionary<int, string> rgVocab = new Dictionary<int, string>();

            for (int i = 0; i < rgstr.Length; i++)
            {
                rgVocab[i] = rgstr[i];
            }

            return rgVocab;
        }

        /// <summary>
        /// Returns the vocabulary size, which should be 32000.
        /// </summary>
        public int Count
        {
            get { return m_rgVocabById.Count; }
        }

        /// <summary>
        /// Returns the beginning of sentence token (BOS = 1).
        /// </summary>
        public int BOS
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the end of sentence token (EOS = 2).
        /// </summary>
        public int EOS
        {
            get { return 2; }  
        }

        /// <summary>
        /// Not used with the LLaMA vocabulary.
        /// </summary>
        /// <param name="str">Not Used</param>
        public void Add(string str)
        {
        }

        /// <summary>
        /// Not used with the LLaMA vocabulary.
        /// </summary>
        /// <returns>The current vocabulary Count is returned.</returns>
        public int Build()
        {
            return Count;
        }

        /// <summary>
        /// Not used with the LLaMA vocabulary.
        /// </summary>
        /// <param name="strData">Not used.</param>
        /// <returns>The current vocabulary Count is returned.</returns>
        public int BuildFromString(string strData)
        {
            return Count;
        }

        /// <summary>
        /// Not used with the LLaMA vocabulary.
        /// </summary>
        /// <param name="rgSrc">Not used.</param>
        /// <returns>Not used.</returns>
        /// <exception cref="NotImplementedException">The NotImplemented exception is thrown when called.</exception>
        public int[] CreateTarget(int[] rgSrc)
        {
            throw new NotImplementedException();
        }

        private string utf8ByteToHex(byte b)
        {
            return "<0x" + b.ToString("X2") + ">";
        }

        private int hextToUtf8Byte(string str)
        {
            str = str.Substring(3, 2);
            str = str.Replace(">", "");
            return Convert.ToInt32(str, 16);
        }

        /// <summary>
        /// Detokenizes the tokens within the 'rgf' array into a string.
        /// </summary>
        /// <param name="rgf">Specifies the token values to be detokenized.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS tokens.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS tokens.</param>
        /// <param name="nStartIdx">Optionally, specifies a starting index (default = 0).</param>
        /// <param name="nCount">Optionally, specifies the number of items to process (default = -1, for all items).</param>
        /// <param name="nPadToken">Optionally, specifies a pad token that is ignored.</param>
        /// <returns>The Detokenized string is returned.</returns>
        public string Detokenize(float[] rgf, bool bIgnoreBos, bool bIgnoreEos, int nStartIdx = 0, int nCount = -1, int? nPadToken = null)
        {
            int nStart = (m_bAddBos) ? nStartIdx + 1 : nStartIdx;
            List<byte> rgUtf8Bytes = new List<byte>();

            if (nCount <= 0)
                nCount = rgf.Count();

            for (int i=nStart; i<nStartIdx + nCount; i++)
            {
                int nTokenId = (int)rgf[i];

                if (nPadToken.HasValue && nTokenId == nPadToken.Value)
                    continue;

                string strToken = m_rgVocabById[nTokenId];

                if (strToken.StartsWith("<0x") && strToken.EndsWith(">"))
                {
                    // Special case.
                    int nUtf8 = hextToUtf8Byte(strToken);
                    rgUtf8Bytes.Add((byte)nUtf8);
                }
                else
                {
                    // Typical case.
                    byte[] rgBytes = Encoding.UTF8.GetBytes(strToken);
                    rgUtf8Bytes.AddRange(rgBytes);
                }
            }

            string strDecoded = Encoding.UTF8.GetString(rgUtf8Bytes.ToArray());
            strDecoded = strDecoded.Replace(m_rgVocabById[29871], " ");

            if (m_bAddPrecedingSpace)
            {
                if (strDecoded.Length > 0 && strDecoded[0] == ' ')
                    strDecoded = strDecoded.Substring(1);
            }

            return strDecoded;
        }

        /// <summary>
        /// Detokenizes a single token into a string.  NOTE: This method is not used with the LLaMA vocabulary for it does not support unicode strings that span multiple tokens.
        /// </summary>
        /// <param name="nTokenId">Specifies the token value to be detokenized.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS tokens.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS tokens.</param>
        /// <returns>The detokenized string is returned.</returns>
        public string Detokenize(int nTokenId, bool bIgnoreBos, bool bIgnoreEos)
        {
            string strToken = m_rgVocabById[nTokenId];
            List<byte> rgUtf8Bytes = new List<byte>();

            if (strToken.StartsWith("<0x") && strToken.EndsWith(">"))
            {
                // Special case.
                int nUtf8 = hextToUtf8Byte(strToken);
                rgUtf8Bytes.Add((byte)nUtf8);
            }
            else
            {
                // Typical case.
                byte[] rgBytes = Encoding.UTF8.GetBytes(strToken);
                rgUtf8Bytes.AddRange(rgBytes);
            }

            string strDecoded = Encoding.UTF8.GetString(rgUtf8Bytes.ToArray());
            strDecoded = strDecoded.Replace(m_rgVocabById[29871], " ");

            return strDecoded;
        }

        /// <summary>
        /// Tokenize a string and return the tokens.
        /// </summary>
        /// <param name="str">Specifies the input string to tokenize. NOTE, input strings with unicode characters are supported.</param>
        /// <param name="bAddBos">Specifies to add the BOS token.</param>
        /// <param name="bAddEos">Specifies to add the EOS token.</param>
        /// <returns>The tokens for the string are returned.</returns>
        public int[] Tokenize(string str, bool bAddBos, bool bAddEos)
        {
            var tokens = Tokenize(str);
            if (bAddBos)
                tokens.Insert(0, BOS);

            if (bAddEos)
                tokens.Add(EOS);

            return tokens.ToArray();
        }

        private List<int> mapCharactersToTokenIds(string strPrompt, bool bAddBos, bool bAddPrecedingSpace)
        {
            List<int> rgTokens = new List<int>();

            // Special beginning of sentence token.
            if (bAddBos)
                rgTokens.Add(BOS);

            // Special 'preceding space' added to beginning of prompt.
            if (bAddPrecedingSpace)
                strPrompt = " " + strPrompt;

            // Special case - spaces are represented as thich underscore _ (id 29871)
            strPrompt = strPrompt.Replace(" ", m_rgVocabById[29871]);
            TextElementEnumerator charEnum = StringInfo.GetTextElementEnumerator(strPrompt);

            // Transform each character to its corresponding token.
            while (charEnum.MoveNext())
            {
                string str = charEnum.GetTextElement();

                if (m_rgVocabByString.ContainsKey(str))
                {
                    rgTokens.Add(m_rgVocabByString[str]);
                }
                else
                {
                    // Special case where token not found and we have to fall back to byte-level tokens.
                    byte[] rgBytes = Encoding.UTF8.GetBytes(str);

                    for (int j = 0; j < rgBytes.Length; j++)
                    {
                        string strHex = utf8ByteToHex(rgBytes[j]);
                        if (m_rgVocabByString.ContainsKey(strHex))
                        {
                            rgTokens.Add(m_rgVocabByString[strHex]);
                        }
                        else
                        {
                            // This is not supposed to happen because the LLaMA vocabulary has a token corresponding to each byte,
                            // but if it does, we fall back to the unknown token.
                            rgTokens.Add(0);
                        }
                    }
                }
            }

            return rgTokens;
        }

        private void addToMergeQueue(Node leftNode, PriorityQueue mergeQueue, string strPrompt)
        {
            string strMergeIdentifyingString = getMergeIdentifyingString(leftNode.TokenId, leftNode.Next.TokenId);
            if (m_rgMerges.ContainsKey(strMergeIdentifyingString))
            {
                int nMergeId = m_rgMerges[strMergeIdentifyingString];
                leftNode.MergePrio = nMergeId + (double)leftNode.OrigPos / strPrompt.Length;
                leftNode.MergeToString = strMergeIdentifyingString.Replace(" ", "");
                mergeQueue.Push(leftNode);
            }
        }

        /// <summary>
        /// Tokenize a string and return the tokens.
        /// </summary>
        /// <param name="bMustExist">Not used.</param>
        /// <param name="strInput">Specifies the input string to tokenize. NOTE, input strings with unicode characters are supported.</param>
        /// <returns>The tokens for the string are returned.</returns>
        public List<int> Tokenize(string strInput, bool bMustExist = true)
        {
            // Initially each character is transformed to a tokenId, later there will be merges of these.
            List<int> rgTokens = mapCharactersToTokenIds(strInput, m_bAddBos, m_bAddPrecedingSpace);
            if (rgTokens.Count < 2)
                return rgTokens;

            PriorityQueue mergeQueue = new PriorityQueue();
            Node firstTokenNode = new Node(rgTokens[0]);
            Node prevTokenNode = firstTokenNode;

            for (int i = 1; i < rgTokens.Count; i++)
            {
                Node currTokenNode = new Node(rgTokens[i], i, prevTokenNode);
                prevTokenNode.Next = currTokenNode;

                addToMergeQueue(prevTokenNode, mergeQueue, strInput);
                prevTokenNode = currTokenNode;
            }

            // Perform merges in priority order.
            while (!mergeQueue.IsEmpty)
            {
                Node leftOfMerge = mergeQueue.Pop();

                // Check that this merge is still possible.
                if (leftOfMerge.Deleted)
                    continue;
                if (leftOfMerge.Next == null)
                    continue;
                if (leftOfMerge.Next.Deleted)
                    continue;

                // Mark leftOfMerge and rightOfMerge as being deleted, because they are being replaced by a merged token.
                leftOfMerge.Deleted = true;
                leftOfMerge.Next.Deleted = true;

                // Its a little more complicated to fix the prev of leftOfMerge.
                if (leftOfMerge.Prev != null)
                {
                    Node oldPrev = leftOfMerge.Prev;
                    // Mark oldPrev as deleted, to avoid erroneous merges later (ref to this node exists in priority queue)
                    oldPrev.Deleted = true;
                    // Replace oldPrev within the linked list with a copy of itself.
                    Node newPrev = new Node(oldPrev);
                    leftOfMerge.Prev = newPrev;
                    // Update linked list reference of 'prev of prev'
                    if (newPrev.Prev != null)
                        newPrev.Prev.Next = newPrev;
                    else
                        firstTokenNode = newPrev; // if 'prev of prev' doesn't exist, then this is the new head of the list.
                }

                // Create node representing merge result
                int nTokenId = m_rgVocabByString[leftOfMerge.MergeToString];
                Node resultOfMerge = new Node(nTokenId, leftOfMerge.OrigPos, leftOfMerge.Prev, leftOfMerge.Next.Next);
                // Consider adding to merge queue: prev--resultOfMerge
                if (resultOfMerge.Prev != null)
                {
                    resultOfMerge.Prev.Next = resultOfMerge;
                    addToMergeQueue(resultOfMerge.Prev, mergeQueue, strInput);
                }
                else
                {
                    // If prev does not exist then this is the firstNode
                    firstTokenNode = resultOfMerge;
                }
                // Consider adding to merge queue: resultOfMerge--next
                if (resultOfMerge.Next != null)
                {
                    resultOfMerge.Next.Prev = resultOfMerge;
                    addToMergeQueue(resultOfMerge, mergeQueue, strInput);
                }
            }

            // Get final tokenIds by traversing the linked list.
            rgTokens.Clear();
            Node node = firstTokenNode;
            while (node != null)
            {
                rgTokens.Add(node.TokenId);
                node = node.Next;
            }

            return rgTokens;
        }
    }

#pragma warning disable 1591, 1587

    class Node /** @private */
    {
        double m_mergePrio = 0;
        string m_strMergeToString = "";
        int m_nOrigPos = 0;
        int m_nTokenId = 0;
        Node m_prev = null;
        Node m_next = null;
        bool m_bDeleted = false;

        public Node(Node node)
        {
            m_nOrigPos = node.m_nOrigPos;
            m_nTokenId = node.m_nTokenId;
            m_prev = node.m_prev;
            m_next = node.m_next;
        }

        public Node(int nTokenId, int nOrigPos = 0, Node prev = null, Node next = null)
        {
            m_nOrigPos = nOrigPos;
            m_nTokenId = nTokenId;
            m_prev = prev;
            m_next = next;
        }

        public bool Deleted
        {
            get { return m_bDeleted; }
            set { m_bDeleted = value; }
        }

        public int TokenId
        {
            get { return m_nTokenId; }
        }

        public double MergePrio
        {
            get { return m_mergePrio; }
            set { m_mergePrio = value; }
        }

        public string MergeToString
        {
            get { return m_strMergeToString; }
            set { m_strMergeToString = value; }
        }

        public int OrigPos
        {
            get { return m_nOrigPos; }
            set { m_nOrigPos = value; }
        }

        public Node Prev
        {
            get { return m_prev; }
            set { m_prev = value; }
        }

        public Node Next
        {
            get { return m_next; }
            set { m_next = value; }
        }

        public override string ToString()
        {
            return m_nOrigPos.ToString() + " " + m_nTokenId.ToString() + " -> " + m_mergePrio.ToString();
        }
    }

    // PriorityQueue implementation is copied from https://stackoverflow.com/a/42919752 with minor refactoring
    class PriorityQueue /** @private */
    {
        List<Node> m_rgHeap = new List<Node>();

        public int Push(Node node)
        {
            m_rgHeap.Add(node);
            shiftUp();
            return m_rgHeap.Count;
        }

        private bool greater(int nIdx1, int nIdx2)
        {
            return m_rgHeap[nIdx1].MergePrio < m_rgHeap[nIdx2].MergePrio;
        }

        private int parent(int nIdx)
        {
            return ((nIdx + 1) >> 1) - 1;
        }

        private int left(int nIdx)
        {
            return (nIdx << 1) + 1;
        }

        private int right(int nIdx)
        {
            return (nIdx + 1) << 1;
        }

        private void swap(int nIdx1, int nIdx2)
        {
            Node node = m_rgHeap[nIdx1];
            m_rgHeap[nIdx1] = m_rgHeap[nIdx2];
            m_rgHeap[nIdx2] = node;
        }

        private void shiftUp()
        {
            int nNode = m_rgHeap.Count - 1;
            while (nNode > 0 && greater(nNode, parent(nNode)))
            {
                swap(nNode, parent(nNode));
                nNode = parent(nNode);
            }
        }

        private void shiftDown()
        {
            int nNode = 0;

            while ((left(nNode) < m_rgHeap.Count && greater(left(nNode), nNode)) ||
                   (right(nNode) < m_rgHeap.Count && greater(right(nNode), nNode)))
            {
                int nMaxChild = (right(nNode) < m_rgHeap.Count && greater(right(nNode), left(nNode))) ? right(nNode) : left(nNode);
                swap(nNode, nMaxChild);
                nNode = nMaxChild;
            }
        }

        public Node Pop()
        {
            Node poppedValue = Peek();
            int nBottom = m_rgHeap.Count - 1;

            if (nBottom > 0)
                swap(0, nBottom);

            m_rgHeap.RemoveAt(m_rgHeap.Count-1);
            shiftDown();

            return poppedValue;
        }

        public Node Peek()
        {
            return m_rgHeap[0];
        }

        public bool IsEmpty
        {
            get { return m_rgHeap.Count == 0; }
        }
    }

#pragma warning restore 1591, 1587
}
