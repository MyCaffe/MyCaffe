using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.IO;

namespace MyCaffe.common
{
    /// <summary>
    /// The BlobCollection contains a list of Blobs.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BlobCollection<T> : IEnumerable<Blob<T>>, IDisposable 
    {
        List<Blob<T>> m_rgBlobs = new List<Blob<T>>();

        /// <summary>
        /// The BlobCollection constructor.
        /// </summary>
        public BlobCollection()
        {
        }

        /// <summary>
        /// Returns the number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgBlobs.Count; }
        }

        /// <summary>
        /// Get/set an item within the collection at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to access.</param>
        /// <returns>The item at the index is returned.</returns>
        public Blob<T> this[int nIdx]
        {
            get { return m_rgBlobs[nIdx]; }
            set { m_rgBlobs[nIdx] = value; }
        }

        /// <summary>
        /// Add a new Blob to the collection.
        /// </summary>
        /// <param name="b">Specifies the Blob to add.</param>
        public void Add(Blob<T> b)
        {
            m_rgBlobs.Add(b);
        }

        /// <summary>
        /// If it exists, remove a Blob from the collection.
        /// </summary>
        /// <param name="b">Specifies the Blob.</param>
        /// <returns>If the Blob is found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Remove(Blob<T> b)
        {
            return m_rgBlobs.Remove(b);
        }

        /// <summary>
        /// Remove a Blob at a given index in the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgBlobs.RemoveAt(nIdx);
        }

        /// <summary>
        /// Remove all items from the collection.
        /// </summary>
        /// <param name="bDispose">Optionally, call Dispose on each item removed.</param>
        public void Clear(bool bDispose = false)
        {
            if (bDispose)
            {
                foreach (Blob<T> b in m_rgBlobs)
                {
                    b.Dispose();
                }
            }

            m_rgBlobs.Clear();
        }

        /// <summary>
        /// Returns whether or not the collection contains a given blob.
        /// </summary>
        /// <param name="blob">Specifies the blob to look for.</param>
        /// <returns>If the blob is in the collection, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Contains(Blob<T> blob)
        {
            return m_rgBlobs.Contains(blob);
        }

        /// <summary>
        /// Find all Blobs in the collection that contain (in part or in whole) the name of a given Blob.
        /// </summary>
        /// <param name="b">Specifies the Blob to look for.</param>
        /// <returns>A new collection of all Blobs that match are returned.</returns>
        public BlobCollection<T> FindRelatedBlobs(Blob<T> b)
        {
            BlobCollection<T> rg = new BlobCollection<T>();

            foreach (Blob<T> b1 in m_rgBlobs)
            {
                if (b1.Name.Contains(b.Name))
                    rg.Add(b1);
            }

            return rg;
        }

        /// <summary>
        /// Get the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<Blob<T>> GetEnumerator()
        {
            return m_rgBlobs.GetEnumerator();
        }

        /// <summary>
        /// Get the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgBlobs.GetEnumerator();
        }

        /// <summary>
        /// Copy the collection and return it as a new collection.
        /// </summary>
        /// <returns>The new collection is returned.</returns>
        public BlobCollection<T> Clone()
        {
            BlobCollection<T> col = new BlobCollection<T>();

            foreach (Blob<T> b in m_rgBlobs)
            {
                col.Add(b.Clone());
            }

            return col;
        }

        /// <summary>
        /// Reshapes all blobs in the collection to the sizes of the source.
        /// </summary>
        /// <param name="src">Specifies the source collection to reshape to.</param>
        public void ReshapeLike(BlobCollection<T> src)
        {
            for (int i = 0; i < m_rgBlobs.Count; i++)
            {
                m_rgBlobs[i].ReshapeLike(src[i]);
            }
        }

        /// <summary>
        /// Copy the data or diff from another BlobCollection into this one.
        /// </summary>
        /// <param name="bSrc">Specifies the src BlobCollection to copy.</param>
        /// <param name="bCopyDiff">Optionally, specifies to copy the diff instead of the data (default = <i>false</i>).</param>
        public void CopyFrom(BlobCollection<T> bSrc, bool bCopyDiff = false)
        {
            if (Count != bSrc.Count)
                throw new Exception("The source and destination should have the same count.");

            for (int i = 0; i < bSrc.Count; i++)
            {
                m_rgBlobs[i].CopyFrom(bSrc[i], bCopyDiff, true);
            }
        }

        /// <summary>
        /// Accumulate the diffs from one BlobCollection into another.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to add the blobs into this collection.</param>
        /// <param name="src">Specifies the source BlobCollection to add into this one.</param>
        /// <param name="bAccumulateDiff">Specifies to accumulate diffs when <i>true</i>, and the data otherwise.</param>
        public void Accumulate(CudaDnn<T> cuda, BlobCollection<T> src, bool bAccumulateDiff)
        {
            for (int i = 0; i < src.Count; i++)
            {
                Blob<T> bSrc = src[i];
                Blob<T> bDst = m_rgBlobs[i];
                int nSrcCount = bSrc.count();
                int nDstCount = bDst.count();

                if (nSrcCount != nDstCount)
                    throw new Exception("The src and dst blobs at index #" + i.ToString() + " have different sizes!");

                if (bAccumulateDiff)
                {
                    if (bSrc.DiffExists && bDst.DiffExists)
                        cuda.add(nSrcCount, bSrc.gpu_diff, bDst.gpu_diff, bDst.mutable_gpu_diff);
                }
                else
                {
                    cuda.add(nSrcCount, bSrc.gpu_data, bDst.gpu_data, bDst.mutable_gpu_data);
                }
            }
        }

        /// <summary>
        /// Set all blob diff to the value specified.
        /// </summary>
        /// <param name="df">Specifies the value to set all blob diff to.</param>
        public void SetDiff(double df)
        {
            for (int i = 0; i < m_rgBlobs.Count; i++)
            {
                m_rgBlobs[i].SetDiff(df);
            }
        }

        /// <summary>
        /// Set all blob data to the value specified.
        /// </summary>
        /// <param name="df">Specifies the value to set all blob data to.</param>
        public void SetData(double df)
        {
            for (int i = 0; i < m_rgBlobs.Count; i++)
            {
                m_rgBlobs[i].SetData(df);
            }
        }

        /// <summary>
        /// Create a new collection of cloned Blobs created by calling MathSub to subtract the Blobs in this collection from another collection.
        /// </summary>
        /// <remarks>
        /// Calculation: Y = Clone(colA) - this
        /// </remarks>
        /// <param name="col">Specifies the collection that this collection will be subtracted from.</param>
        /// <param name="bSkipFirstItem">Specifies whether or not to skip the first item.</param>
        /// <returns></returns>
        public BlobCollection<T> MathSub(BlobCollection<T> col, bool bSkipFirstItem)
        {
            if (col.Count != m_rgBlobs.Count)
                throw new Exception("The input blob collection must have the same count at this blob collection!");

            BlobCollection<T> colOut = new BlobCollection<T>();

            for (int i = 0; i < m_rgBlobs.Count; i++)
            {
                if (i > 0 || !bSkipFirstItem)
                    colOut.Add(m_rgBlobs[i].MathSub(col[i]));
            }

            return colOut;
        }

        /// <summary>
        /// Create a new collection of cloned Blobs created by calling MathAdd to add the Blobs in this collection to another collection.
        /// </summary>
        /// <remarks>
        /// Calculation: Y = Clone(colA) * dfScale + this
        /// </remarks>
        /// <param name="colA">Specifies the collection that will be cloned.</param>
        /// <param name="dfScale">Specifies the scale that will be applied to the clone of this collection</param>
        /// <param name="bSkipFirstItem">Specifies whether or not to skip the first item.</param>
        /// <returns></returns>
        public BlobCollection<T> MathAdd(BlobCollection<T> colA, double dfScale, bool bSkipFirstItem)
        {
            if (colA.Count != m_rgBlobs.Count)
                throw new Exception("The input blob collection must have the same count at this blob collection!");

            BlobCollection<T> colOut = new BlobCollection<T>();
            T fScale = (T)Convert.ChangeType(dfScale, typeof(T));

            for (int i = 0; i < m_rgBlobs.Count; i++)
            {
                if (i > 0 || !bSkipFirstItem)
                    colOut.Add(m_rgBlobs[i].MathAdd(colA[i], fScale));
            }

            return colOut;
        }

        /// <summary>
        /// Create a new collection of cloned Blobs created by calling MathDif to divide the Blobs in this collection with a scalar.
        /// </summary>
        /// <param name="dfVal">Specifies the divisor.</param>
        /// <param name="bSkipFirstItem">Specifies whether or not to skip the first item.</param>
        /// <returns></returns>
        public BlobCollection<T> MathDiv(double dfVal, bool bSkipFirstItem)
        {
            BlobCollection<T> colOut = new BlobCollection<T>();
            T fVal = (T)Convert.ChangeType(dfVal, typeof(T));

            for (int i = 0; i < m_rgBlobs.Count; i++)
            {
                if (i > 0 || !bSkipFirstItem)
                    colOut.Add(m_rgBlobs[i].MathDiv(fVal));
            }

            return colOut;
        }

        /// <summary>
        /// Save the collection to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="bData">Specifies whether or not to save the data.</param>
        /// <param name="bDiff">Specifies whether or not to save the diff.</param>
        public void Save(BinaryWriter bw, bool bData, bool bDiff)
        {
            bw.Write(m_rgBlobs.Count);

            foreach (Blob<T> b in m_rgBlobs)
            {
                b.Save(bw, bData, bDiff);
            }
        }

        /// <summary>
        /// Loads a new collection from a binary reader.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use for the Cuda connection.</param>
        /// <param name="log">Specifies the Log to use for output.</param>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bData">Specifies whether or not to read the data.</param>
        /// <param name="bDiff">Specifies whether or not to read the diff.</param>
        /// <returns></returns>
        public static BlobCollection<T> Load(CudaDnn<T> cuda, Log log, BinaryReader br, bool bData, bool bDiff)
        {
            BlobCollection<T> col = new BlobCollection<T>();
            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                col.Add(Blob<T>.Load(cuda, log, br, bData, bDiff));
            }

            return col;
        }

        /// <summary>
        /// Share the first Blob found with the same name as the given Blob.
        /// </summary>
        /// <param name="b">Specifies the Blob that will share the found Blob.</param>
        /// <param name="rgMinShape">Specifies the minimum shape required to share.</param>
        /// <param name="bThrowExceptions">Specifies whether or not th throw an Exception if the sizes do not match.</param>
        /// <returns></returns>
        public bool Share(Blob<T> b, List<int> rgMinShape, bool bThrowExceptions)
        {
            int nCount = 0;

            if (rgMinShape != null && rgMinShape.Count > 0)
            {
                nCount = rgMinShape[0];
                for (int i = 1; i < rgMinShape.Count; i++)
                {
                    nCount *= rgMinShape[i];
                }
            }

            foreach (Blob<T> blobShare in m_rgBlobs)
            {
                if (blobShare.Name == b.Name)
                {
                    if (nCount > 0)
                    {
                        if (blobShare.count() < nCount)
                        {
                            if (bThrowExceptions)
                                throw new Exception("The blob to be shared is smaller that the expected minimum count!");

                            return false;
                        }
                    }

                    b.Share(blobShare);
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Release all resource used by the collection and its Blobs.
        /// </summary>
        public void Dispose()
        {
            foreach (Blob<T> b in m_rgBlobs)
            {
                b.Dispose();
            }
        }
    }
}
