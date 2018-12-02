using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.param;
using MyCaffe.basecode;

namespace MyCaffe.common
{
    /// <summary>
    /// The WorkspaceArgs are passed to both the Layer::OnSetWorkspace and Layer::OnGetWorkspace events.
    /// </summary>
    /// <remarks>
    /// These events allow for sharing workspace GPU memory among layers thus conserving overall GPU memory usage.
    /// </remarks>
    public class WorkspaceArgs : EventArgs
    {
        long m_lWorkspaceSizeInBytes = 0;
        long m_hWorkspaceData = 0;  // underlying storage.

        /// <summary>
        /// The WorkspaceArgs constructor.
        /// </summary>
        /// <param name="hData">Specifies a handle to the GPU memory.</param>
        /// <param name="lSize">Specifies the size of the workspace memory (in bytes).</param>
        public WorkspaceArgs(long hData, long lSize)
        {
            m_hWorkspaceData = hData;
            m_lWorkspaceSizeInBytes = lSize;
        }

        /// <summary>
        /// Get/set the handle to workspace data in GPU memory.
        /// </summary>
        public long Data
        {
            get { return m_hWorkspaceData; }
            set { m_hWorkspaceData = value; }
        }

        /// <summary>
        /// Get/set the size of the workspace memory (in bytes).
        /// </summary>
        public long Size
        {
            get { return m_lWorkspaceSizeInBytes; }
            set { m_lWorkspaceSizeInBytes = value; }
        }
    }

    /// <summary>
    /// The GetWorkBlobArgs are passed to the Layer::OnGetWorkBlob event which is supported for debugging only.
    /// </summary>
    /// <typeparam name="T">Specifies the default type.</typeparam>
    public class GetWorkBlobArgs<T> : EventArgs
    {
        Blob<T> m_work = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public GetWorkBlobArgs()
        {
        }

        /// <summary>
        /// Specifies the blob.
        /// </summary>
        public Blob<T> Blob
        {
            get { return m_work; }
            set { m_work = value; }
        }
    }

    /// <summary>
    /// The TestArgs are passed to the Solver::OnTest event.
    /// </summary>
    /// <remarks>
    /// The Solver:OnTest event allows for overriding the Solver::Test functionality.
    /// </remarks>
    public class TestArgs : EventArgs
    {
        int m_nIterationOverride = -1;
        int m_nTestNetID = 0;
        double m_dfAccuracy = 0;
        
        /// <summary>
        /// The TestArgs constructor.
        /// </summary>
        /// <param name="nIterationOverride">When greater than 0, specifies a testing iteration override, otherwise the value is ignored.</param>
        /// <param name="nTestNetID">Specifies the test Net that the Solver would like to test.</param>
        public TestArgs(int nIterationOverride, int nTestNetID)
        {
            m_nIterationOverride = nIterationOverride;
            m_nTestNetID = nTestNetID;
        }

        /// <summary>
        /// Returns the testing iteration override.  When set to -1, the solver description test iteration is used.
        /// </summary>
        public int IterationOverride
        {
            get { return m_nIterationOverride; }
        }

        /// <summary>
        /// Returns the test Net identifier of the Solver test Net to run.
        /// </summary>
        public int TestNetID
        {
            get { return m_nTestNetID; }
        }

        /// <summary>
        /// Get/set the accuracy for the test run.  When overriding the testing, the override should set the accuracy value.
        /// </summary>
        public double Accuracy
        {
            get { return m_dfAccuracy; }
            set { m_dfAccuracy = value; }
        }
    }

    /// <summary>
    /// Specifies the TestingIterationArgs sent to the Solver::OnTestingIteration, which is called at the end of a testing cycle.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TestingIterationArgs<T> : EventArgs
    {
        double m_dfAccuracy;
        int m_nIteration;
        double m_dfMsTiming = 0;

        /// <summary>
        /// The TestingIterationArgs constructor.
        /// </summary>
        /// <param name="nIteration">Specifies the iteration of the test cycle.</param>
        /// <param name="dfAccuracy">Specifies the accuracy of the test cycle.</param>
        /// <param name="dfMsTiming">Specifies the timing (in ms.) of the test cycle.</param>
        public TestingIterationArgs(int nIteration, double dfAccuracy, double dfMsTiming)
        {
            m_dfAccuracy = dfAccuracy;
            m_nIteration = nIteration;
            m_dfMsTiming = dfMsTiming;
        }

        /// <summary>
        /// Return the accuracy of the test cycle.
        /// </summary>
        public double Accuracy
        {
            get { return m_dfAccuracy; }
        }

        /// <summary>
        /// Return the iteration of the test cycle.
        /// </summary>
        public int Iteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Return the timing of the test cycle in ms.
        /// </summary>
        public double Timing
        {
            get { return m_dfMsTiming; }
        }
    }

    /// <summary>
    /// The TrainingIterationArgs is sent to the Solver::OnTrainingIteration event that fires at the end of a training cycle.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TrainingIterationArgs<T> : TestingIterationArgs<T>
    {
        double m_dfLoss;
        double m_dfSmoothedLoss;
        double m_dfBestSmoothedLoss;
        bool m_bWeightsUpdated = false;
        string m_strActiveLabelCounts = "";
        double m_dfLearningRate = 0;
        DebugInformation<T> m_dbgInfo = null;

        /// <summary>
        /// The TrainingIterationArgs constructor.
        /// </summary>
        /// <param name="nIteration">Specifies the iteration of the training cycle.</param>
        /// <param name="dfAccuracy">Specifies the last accuracy recieved during the training cycle's last testing cycle.</param>
        /// <param name="dfLoss">Specifies the loss of the training cycle.</param>
        /// <param name="dfSmoothedLoss">Specifies the averaged loss of the training cycle.</param>
        /// <param name="dfBestSmoothedLoss">Specifies the best smoothed loss observed so far during the training.</param>
        /// <param name="bWeightsUpdated">Specifies whether or not the weights have been updated.</param>
        /// <param name="strActiveLabelCounts">Specifies the current active label counts observed.</param>
        /// <param name="dfLearningRate">Specifies the current learning rate.</param>
        /// <param name="dfMsTiming">Specifies the timing of the training cycle.</param>
        /// <param name="dbgInfo">Optionally, specifies the DebugInformation of the training cycle.  This value is set when Solver::EnableBlobDebugging == <i>true</i>.</param>
        public TrainingIterationArgs(int nIteration, double dfAccuracy, double dfLoss, double dfSmoothedLoss, double dfBestSmoothedLoss, bool bWeightsUpdated, string strActiveLabelCounts, double dfLearningRate, double dfMsTiming, DebugInformation<T> dbgInfo = null)
            : base(nIteration, dfAccuracy, dfMsTiming)
        {
            m_dfLoss = dfLoss;
            m_dfSmoothedLoss = dfSmoothedLoss;
            m_dfBestSmoothedLoss = dfBestSmoothedLoss;
            m_bWeightsUpdated = bWeightsUpdated;
            m_strActiveLabelCounts = strActiveLabelCounts;
            m_dfLearningRate = dfLearningRate;
            m_dbgInfo = dbgInfo;
        }

        /// <summary>
        /// Returns the loss of the training cycle.
        /// </summary>
        public double Loss
        {
            get { return m_dfLoss; }
        }

        /// <summary>
        /// Retunrs the average loss after the training cycle.
        /// </summary>
        public double SmoothedLoss
        {
            get { return m_dfSmoothedLoss; }
        }

        /// <summary>
        /// Returns the best smoothed loss observed during the training.
        /// </summary>
        public double BestSmoothedLoss
        {
            get { return m_dfBestSmoothedLoss; }
        }

        /// <summary>
        /// Returns whether or not the weights have been updated.
        /// </summary>
        public bool WeightsUpdated
        {
            get { return m_bWeightsUpdated; }
        }

        /// <summary>
        /// Returns the current active label counts as a string.
        /// </summary>
        public string ActiveLabelCounts
        {
            get { return m_strActiveLabelCounts; }
        }

        /// <summary>
        /// Return the current learning rate.
        /// </summary>
        public double LearningRate
        {
            get { return m_dfLearningRate; }
        }

        /// <summary>
        /// Returns the DebugInformation (if any).  The DebugInformation is set to null when Solver::EnableBlobDebugging == <i>false</i>.
        /// </summary>
        public DebugInformation<T> DebugInformation
        {
            get { return m_dbgInfo; }
        }
    }

    /// <summary>
    /// The GetBytesArgs is passed along to the SnapshotArgs::OnGetWeights and SnapshotArgs::OnGetState events.
    /// </summary>
    public class GetBytesArgs : EventArgs
    {
        byte[] m_rgBytes = null;

        /// <summary>
        /// The GetBytesArgs constructor.
        /// </summary>
        public GetBytesArgs()
        {
        }

        /// <summary>
        /// Get/set the data as an array of bytes.
        /// </summary>
        public byte[] Data
        {
            get { return m_rgBytes; }
            set { m_rgBytes = value; }
        }
    }

    /// <summary>
    /// The SnapshotArgs is sent to the Solver::OnSnapshot event which fires each time the Solver::Snapshot method is called.
    /// </summary>
    public class SnapshotArgs : EventArgs
    {
        byte[] m_rgWeights = null;
        byte[] m_rgState = null;
        double m_dfAccuracy = 0;
        double m_dfError = 0;
        int m_nIteration = 0;
        SNAPSHOT_WEIGHT_UPDATE_METHOD m_favor = SNAPSHOT_WEIGHT_UPDATE_METHOD.FAVOR_ACCURACY;
        bool m_bIncludeWeights = true;
        bool m_bIncludeState = false;
        bool m_bSingleStep = false;
        bool m_bForced = false;
        bool m_bScheduled = true;


        /// <summary>
        /// Specifies the OnGetWeights event which fires when the SnapshotArgs::UpdateWeights method is called.
        /// </summary>
        /// <remarks>
        /// The Solver hooks into these events so that it can access the training Net weights and return them to the caller of the SnapshotArgs::UpdateWeights method.
        /// </remarks>
        public event EventHandler<GetBytesArgs> OnGetWeights;
        /// <summary>
        /// Specifies the OnGetState event which fires when the SnapshotArgs::UpdateState method is called.
        /// </summary>
        /// <remarks>
        /// The Solver hooks into these events so that it can access the Solver state and return it to the caller of the SnapshotArgs::UpdateState method.
        /// </remarks>
        public event EventHandler<GetBytesArgs> OnGetState;

        /// <summary>
        /// The SnapshotArgs constructor.
        /// </summary>
        /// <param name="rgState">Specifies the current Solver state as an array of bytes.</param>
        /// <param name="rgWeights">Specifies the current training Net weights as an array of bytes.</param>
        /// <param name="dfAccuracy">Specifies the last accuracy observed in the training Net.</param>
        /// <param name="dfError">Specifies the last error observed in the training Net.</param>
        /// <param name="nIteration">Specifies the current iteration of training.</param>
        /// <param name="favor">Specifies whether to favor the error value or the accuracy value when deciding whether or not a snapshot should take place.</param>
        public SnapshotArgs(byte[] rgState, byte[] rgWeights, double dfAccuracy, double dfError, int nIteration, SNAPSHOT_WEIGHT_UPDATE_METHOD favor)
        {
            m_rgState = rgState;
            m_rgWeights = rgWeights;
            m_dfAccuracy = dfAccuracy;
            m_dfError = dfError;
            m_nIteration = nIteration;
            m_favor = favor;
        }

        /// <summary>
        /// Retrieves the updated Solver state as an array of bytes.
        /// </summary>
        /// <returns>The state is returned as an array of bytes.</returns>
        public byte[] UpdateState()
        {
            if (OnGetState != null)
            {
                GetBytesArgs args = new common.GetBytesArgs();
                OnGetState(this, args);
                m_rgState = args.Data;
                return m_rgState;
            }

            return null;
        }

        /// <summary>
        /// Retrieves the updated training Net weights as an array of bytes.
        /// </summary>
        /// <returns>The training Net weights are returned as an array of bytes.</returns>
        public byte[] UpdateWeights()
        {
            if (OnGetWeights != null)
            {
                GetBytesArgs args = new common.GetBytesArgs();
                OnGetWeights(this, args);
                m_rgWeights = args.Data;
                return m_rgWeights;
            }

            return null;
        }

        /// <summary>
        /// Get/set the Solver State.
        /// </summary>
        public byte[] State
        {
            get { return m_rgState; }
            set { m_rgState = value; }
        }

        /// <summary>
        /// Get/set the Weights.
        /// </summary>
        public byte[] Weights
        {
            get { return m_rgWeights; }
            set { m_rgWeights = value; }
        }

        /// <summary>
        /// Returns the last observed Solver accuracy for the current training session.
        /// </summary>
        public double Accuracy
        {
            get { return m_dfAccuracy; }
        }

        /// <summary>
        /// Returns the last observed Solver error for the current training session.
        /// </summary>
        public double Error
        {
            get { return m_dfError; }
        }

        /// <summary>
        /// Returns the current iteration of the current training session.
        /// </summary>
        public int Iteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Specifies whether to favor the error, the accuracy or both when deciding whether a snapshot should take place.
        /// </summary>
        public SNAPSHOT_WEIGHT_UPDATE_METHOD Favor
        {
            get { return m_favor; }
        }

        /// <summary>
        /// Get/set whether or not to include the weights in the snapshot.
        /// </summary>
        public bool IncludeWeights
        {
            get { return m_bIncludeWeights; }
            set { m_bIncludeWeights = value; }
        }

        /// <summary>
        /// Get/set whether or not to include the Solver state in the snapshot.
        /// </summary>
        public bool IncludeState
        {
            get { return m_bIncludeState; }
            set { m_bIncludeState = value; }
        }

        /// <summary>
        /// Get/set the Solver single step.
        /// </summary>
        public bool SingleStep
        {
            get { return m_bSingleStep; }
            set { m_bSingleStep = value; }
        }

        /// <summary>
        /// Get/set whether or not the snapshot was forced or not.
        /// </summary>
        public bool Forced
        {
            get { return m_bForced; }
            set { m_bForced = value; }
        }

        /// <summary>
        /// Get/set whether or not the snapshot is a regular scheduled snapshot (e.g. not an improved accuracy or forced snapshot)
        /// </summary>
        public bool Scheduled
        {
            get { return m_bScheduled; }
            set { m_bScheduled = value; }
        }
    }

    /// <summary>
    /// The CustomForwardBackArgs provide the arguments to the OnCustomForwardBack event within the Solver Step function.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class CustomForwardBackArgs<T> : EventArgs
    {
        Net<T> m_net;
        TRAIN_STEP m_step;
        bool m_bFwdPassNanFree = true;
        double m_dfLocalLoss = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="net">Specifies the training network.</param>
        /// <param name="step">Specifies whether or not to step the operation.</param>
        public CustomForwardBackArgs(Net<T> net, TRAIN_STEP step)
        {
            m_net = net;
            m_step = step;
        }

        /// <summary>
        /// Returns the training network.
        /// </summary>
        public Net<T> net
        {
            get { return m_net; }
        }

        /// <summary>
        /// Returns whether or not to step the operation.
        /// </summary>
        public TRAIN_STEP step
        {
            get { return m_step; }
        }

        /// <summary>
        /// Get/set whether or a NAN was detected in the forward pass.
        /// </summary>
        public bool FwdPassNanFree
        {
            get { return m_bFwdPassNanFree; }
            set { m_bFwdPassNanFree = value; }
        }

        /// <summary>
        /// Get/set the local loss of the pass.
        /// </summary>
        public double LocalLoss
        {
            get { return m_dfLocalLoss; }
            set { m_dfLocalLoss = value; }
        }
    }

    /// <summary>
    /// The ForwardArgs are passed to the OnForward event of the EventLayer.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ForwardArgs<T>
    {
        BlobCollection<T> m_colTop;
        BlobCollection<T> m_colBottom;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="colBottom">Specifies the bottom blobs.</param>
        /// <param name="colTop">Specifies the top blobs.</param>
        public ForwardArgs(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_colTop = colTop;
            m_colBottom = colBottom;
        }

        /// <summary>
        /// Returns the bottom blobs.
        /// </summary>
        public BlobCollection<T> BottomVec
        {
            get { return m_colBottom;  }
        }

        /// <summary>
        /// Returns the top blobs.
        /// </summary>
        public BlobCollection<T> TopVec
        {
            get { return m_colTop; }
        }
    }

    /// <summary>
    /// The BackwardArgs are passed to the OnBackward event of the EventLayer.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BackwardArgs<T> : ForwardArgs<T>
    {
        List<bool> m_rgbPropagateDown;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="colBottom">Specifies the bottom blobs.</param>
        /// <param name="colTop">Specifies the top blobs.</param>
        public BackwardArgs(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
            : base(colBottom, colTop)
        {
            m_rgbPropagateDown = rgbPropagateDown;
        }

        /// <summary>
        /// Returns the list on whether or not to propagate down.
        /// </summary>
        public List<bool> PropagateDown
        {
            get { return m_rgbPropagateDown; }
        }
    }

    /// <summary>
    /// The GradientsReadyArgs is sent to the Solver::OnGradientsReady event which fires at the end of each Solver::Step. 
    /// </summary>
    /// <remarks>
    /// The Solver::OnGradientReady event is used in multi-GPU training.
    /// </remarks>
    public class GradientsReadyArgs : EventArgs 
    {
        /// <summary>
        /// The GradientReadyArgs constructor.
        /// </summary>
        public GradientsReadyArgs()
        {
        }
    }

    /// <summary>
    /// The GetIterationArgs is sent bubbled up to the solver when a layer needs to know
    /// the curret training iteration.
    /// </summary>
    public class GetIterationArgs : EventArgs
    {
        int m_nIteration = 0;
        Phase m_phase = Phase.TRAIN;

        /// <summary>
        /// The constructor.
        /// </summary>
        public GetIterationArgs()
        {
        }

        /// <summary>
        /// The SetIteration method is used to set the iteration and the phase.
        /// </summary>
        /// <param name="p">Specifies the phase associated with the iteration.</param>
        /// <param name="nIteration">Specifies the iteration.</param>
        public void SetIteration(Phase p, int nIteration)
        {
            m_phase = p;
            m_nIteration = nIteration;
        }

        /// <summary>
        /// Returns the iteration.
        /// </summary>
        public int Iteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Returns the phase.
        /// </summary>
        public Phase CurrentPhase
        {
            get { return m_phase; }
        }
    }
}
