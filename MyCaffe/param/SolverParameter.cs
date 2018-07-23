using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// The SolverParameter is a parameter for the solver, specifying the train and test networks.
    /// </summary>
    /// <remarks>
    /// Exactly one train net must be specified using one of the following fields:
    ///   train_net_param, train_net, net_param, net
    /// 
    /// One or more of the test nets may be specified using any of the following fields:
    ///   test_net_param, test_net, net_param, net
    ///   
    /// If more than one test net field is specified (e.g., both net and test_net
    /// are specified), they will be evaluated in the field order given above:
    ///   (1) test_net_param, (2) test_net, (3) net_param/net
    ///   
    /// A test_iter must be specified for each test_net.
    /// A test_level and/or test_stage may also be specified for each test_net.
    /// </remarks>
    public class SolverParameter : BaseParameter
    {
        NetParameter m_paramNet = null;
        NetParameter m_paramTrainNet = null;
        List<NetParameter> m_rgTestNets = new List<NetParameter>();
        NetState m_stateTrain = null;
        List<NetState> m_rgStateTest = new List<NetState>();
        List<int> m_rgTestIter = new List<int>() { 300 };
        int m_nTestInterval = 1000;
        bool m_bTestComputeLoss = false;
        bool m_bTestInitialization = true;
        double m_dfBaseLR = 0.01;
        int m_nDisplay = 0;
        int m_nAverageLoss = 1;
        int m_nMaxIter = 500000;
        int m_nIterSize = 1;
        string m_strLrPolicy = "step";
        double m_dfGamma = 0.1;
        double m_dfPower;
        double m_dfMomentum = 0.0;
        double m_dfWeightDecay = 0.0005;
        string m_strRegularizationType = "L2";
        int m_nStepSize = 100000;
        List<int> m_rgnStepValue = new List<int>();
        double m_dfClipGradients = -1;
        int m_nSnapshot = 10000;
        string m_strSnapshotPrefix = "";
        bool m_bSnapshotDiff = false;
        SnapshotFormat m_snapshotFormat = SnapshotFormat.BINARYPROTO;
        int m_nDeviceID = 1;
        long m_lRandomSeed = -1;
        SolverType m_solverType = SolverType.SGD;
        double m_dfDelta = 1e-8;
        double m_dfMomentum2 = 0.999;
        double m_dfRmsDecay = 0.95;
        bool m_bDebugInfo = false;
        bool m_bSnapshotAfterTrain = false;
        string m_strCustomTrainer = null;
        string m_strCustomTrainerProperties = null;
        bool m_bOutputAverageResults = false;
        bool m_bSnapshotIncludeWeights = true;
        bool m_bSnapshotIncludeState = true;

        /// <summary>
        /// Defines the format of each snapshot.
        /// </summary>
        public enum SnapshotFormat
        {
            /// <summary>
            /// Save snapshots in the binary prototype format.
            /// </summary>
            BINARYPROTO = 1
        }

        /// <summary>
        /// Defines the type of solver.
        /// </summary>
        public enum SolverType
        {
            /// <summary>
            /// Use Stochastic Gradient Descent solver with momentum updates weights by a linear combination of the negative gradient and the previous weight update.
            /// </summary>
            /// <remarks>
            /// @see [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) Wikipedia.
            /// </remarks>
            SGD = 0,
            /// <summary>
            /// Use Nesterov's accelerated gradient, similar to SGD, but error gradient is computed on the weights with added momentum.
            /// </summary>
            /// <remarks>
            /// @see [Lecture 6c The momentum method](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) by Hinton, Geoffrey and Srivastava, Nitish and Swersky, Kevin, 2012.
            /// @see [Nesterov's Accelerated Gradient and Momentum as approximations to Regularised Update Descent](https://arxiv.org/abs/1607.01981) by Botev, Alexandar and Lever, Guy and Barber, David, 2016.
            /// </remarks>
            NESTEROV = 1,
            /// <summary>
            /// Use Gradient based optimization like SGD that tries to find rarely seen features
            /// </summary>
            /// <remarks>
            /// @see [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) by Duchi, John and Hazan, Elad, and Singer, Yoram, 2011.
            /// </remarks>
            ADAGRAD = 2,
            /// <summary>
            /// Use RMS Prop gradient based optimization like SGD.
            /// </summary>
            /// <remarks>
            /// @see [Lecture 6e	rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) by Tieleman and Hinton, 2012,
            /// @see [RMSProp and equilibrated adaptive learning rates for non-convex optimization](https://arxiv.org/abs/1502.04390v1) by Dauphin, Yann N. and de Vries, Harm and Chung, Junyoung and Bengio, Yoshua, 2015.
            /// </remarks>
            RMSPROP = 3,
            /// <summary>
            /// Use AdaDelta gradient based optimization like SGD.
            /// </summary>
            /// <remarks>
            /// See [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701) by Zeiler, Matthew D., 2012.
            /// </remarks>
            ADADELTA = 4,
            /// <summary>
            /// Use Adam gradient based optimization like SGD that includes 'adaptive momentum estimation' and can be thougth of as a generalization of AdaGrad.
            /// </summary>
            /// <remarks>
            /// @see [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v9) by Kingma, Diederik P. and Ba, Jimmy, 2014.
            /// </remarks>
            ADAM = 5,
            _MAX = 6
        }

        /// <summary>
        /// Defines the learning rate policy to use.
        /// </summary>
        public enum LearningRatePolicyType
        {
            /// <summary>
            /// Use a fixed learning rate which always returns base_lr.
            /// </summary>
            FIXED,
            /// <summary>
            /// Use a stepped learning rate which returns @f$ base_lr * gamma ^ {floor{iter/step}} @f$
            /// </summary>
            STEP,
            /// <summary>
            /// Use an exponential learning rate which returns @f$ base_lr * gamma ^ {iter} @f$
            /// </summary>
            EXP,
            /// <summary>
            /// Use an inverse learning rate which returns @f$ base_lr * {1 + gamma * iter}^{-power} @f$
            /// </summary>
            INV,
            /// <summary>
            /// Use a multi-step learning rate which is similar to INV, but allows for non-uniform steps defined by stepvalue.
            /// </summary>
            MULTISTEP,
            /// <summary>
            /// Use a polynomial learning rate where the effective learning rate follows a polynomial decay, to be zero by the max_iter.  Returns @f$ base_lr * {1 - iter/max_iter}^{power} @f$
            /// </summary>
            POLY,
            /// <summary>
            /// Use a sigmoid learning rate where the effective learning rate follows a sigmoid decay.  Returns @f$ base_lr * {1/{1 + exp{-gamma * {iter - stepsize}}}} @f$
            /// </summary>
            SIGMOID
        }

        /// <summary>
        /// Defines the regularization type.
        /// </summary>
        public enum RegularizationType
        {
            /// <summary>
            /// Specifies L1 regularization.
            /// </summary>
            L1,
            /// <summary>
            /// Specifies L2 regularization.
            /// </summary>
            L2
        }

        /// <summary>
        /// The SolverParameter constructor.
        /// </summary>
        public SolverParameter()
            : base()
        {
        }

        /// <summary>
        /// Creates a new copy of the SolverParameter.
        /// </summary>
        /// <returns>A new instance of the SolverParameter is returned.</returns>
        public SolverParameter Clone()
        {
            SolverParameter p = SolverParameter.FromProto(ToProto("clone"));

            return p;
        }

        /// <summary>
        /// Specifies to average loss results before they are output - this can be faster when there are a lot of results in a cycle.
        /// </summary>
        [Description("Specifies to average loss results before they are output - this can be faster when there are a lot of results in a cycle.")]
        public bool output_average_results
        {
            get { return m_bOutputAverageResults; }
            set { m_bOutputAverageResults = value; }
        }

        /// <summary>
        /// Specifies the custom trainer (if any) - this is an optional setting used by exteral software to 
        /// provide a customized training process.
        /// </summary>
        [Description("Specifies the custom trainer (if any) used by an external process to provide customized training.")]
        public string custom_trainer
        {
            get { return m_strCustomTrainer; }
            set { m_strCustomTrainer = value; }
        }

        /// <summary>
        /// Specifies the custom trainer properties (if any) - this is an optional setting used by exteral software to 
        /// provide the propreties for a customized training process.
        /// </summary>
        [Description("Specifies the custom trainer properties (if any) used by an external process to provide the properties for a customized training.")]
        public string custom_trainer_properties
        {
            get { return m_strCustomTrainerProperties; }
            set { m_strCustomTrainerProperties = value; }
        }

        /// <summary>
        /// Inline train net param, possibly combined with one or more test nets.
        /// </summary>
        [Browsable(false)]
        public NetParameter net_param
        {
            get { return m_paramNet; }
            set { m_paramNet = value; }
        }

        /// <summary>
        /// Inline train net param, possibly combined with one or more test nets.
        /// </summary>
        [Browsable(false)]
        public NetParameter train_net_param
        {
            get { return m_paramTrainNet; }
            set { m_paramTrainNet = value; }
        }

        /// <summary>
        /// Inline test net params.
        /// </summary>
        [Browsable(false)]
        public List<NetParameter> test_net_param
        {
            get { return m_rgTestNets; }
            set { m_rgTestNets = value; }
        }

        /// <summary>
        /// The states for the train/test nets.  Must be unspecified or
        /// specified once per net.
        /// </summary>
        /// <remarks>
        /// By default, all states will have solver = true;
        /// train_state will have phase = TRAIN,
        /// and all test_state's will have phase = TESET.
        /// Other defaults are set according to NetState defaults.
        /// </remarks>
        [Browsable(false)]
        public NetState train_state
        {
            get { return m_stateTrain; }
            set { m_stateTrain = value; }
        }

        /// <summary>
        /// The states for the train/test nets.  Must be unspecified or
        /// specified once per net.
        /// </summary>
        /// <remarks>
        /// By default, all states will have solver = true;
        /// train_state will have phase = TRAIN,
        /// and all test_state's will have phase = TESET.
        /// Other defaults are set according to NetState defaults.
        /// </remarks>
        [Browsable(false)]
        public List<NetState> test_state
        {
            get { return m_rgStateTest; }
            set { m_rgStateTest = value; }
        }

        /// <summary>
        /// The number of iterations for each test.
        /// </summary>
        [Category("Iterations")]
        [Description("Specifies the number of iterations for each test.")]
        public List<int> test_iter
        {
            get { return m_rgTestIter; }
            set { m_rgTestIter = value; }
        }

        /// <summary>
        /// The number of iterations between two testing phases.
        /// </summary>
        [Category("Iterations")]
        [Description("Specifies the number of iterations between two testing phases.")]
        public int test_interval
        {
            get { return m_nTestInterval; }
            set { m_nTestInterval = value; }
        }

        /// <summary>
        /// Test the compute loss.
        /// </summary>
        [Description("Specifies whether or not to test the compute loss.")]
        public bool test_compute_loss
        {
            get { return m_bTestComputeLoss; }
            set { m_bTestComputeLoss = value; }
        }

        /// <summary>
        /// If true, run an initial test pass before the first iteration,
        /// ensuring memory availability and printing the starting value of the loss.
        /// </summary>
        [Category("Iterations")]
        [Description("If true, run an initial test pass before the first iteration, ensuring memory availability and printing the starting value of the loss.")]
        public bool test_initialization
        {
            get { return m_bTestInitialization; }
            set { m_bTestInitialization = value; }
        }

        /// <summary>
        /// The base learning rate.
        /// </summary>
        [Description("Specifies the base learning rate.")]
        public double base_lr
        {
            get { return m_dfBaseLR; }
            set { m_dfBaseLR = value; }
        }

        /// <summary>
        /// The number of iterations between displaying info.  If display = 0, no info
        /// will be displayed.
        /// </summary>
        [Category("Iterations")]
        [Description("Specifies the number of iterations between displaying information.  If display == 0, no information will be displayed.")]
        public int display
        {
            get { return m_nDisplay; }
            set { m_nDisplay = value; }
        }

        /// <summary>
        /// Display the loss averaged over the last average_loss iterations.
        /// </summary>
        [Description("Specifies the loss averaged over the last 'average_loss' iterations.")]
        public int average_loss
        {
            get { return m_nAverageLoss; }
            set { m_nAverageLoss = value; }
        }

        /// <summary>
        /// The maximum number of iterations.
        /// </summary>
        [Category("Iterations")]
        [Description("Specifies the maximum number of iterations.")]
        public int max_iter
        {
            get { return m_nMaxIter; }
            set { m_nMaxIter = value; }
        }

        /// <summary>
        /// Accumulate gradients over 'iter_size' x 'batch_size' instances.
        /// </summary>
        [Category("Iterations")]
        [Description("Specifies to accumulate gradients over 'iter_size' x 'batch_size' instances.")]
        public int iter_size
        {
            get { return m_nIterSize; }
            set { m_nIterSize = value; }
        }

        /// <summary>
        /// The learning rate decay policy.
        /// </summary>
        /// <remarks>
        /// The currently implemented learning rate policies are as follows:
        ///    - fixed: always return @f$ base_lr @f$.
        ///    - step: return @f$ base_lr * gamma ^ {floor{iter / step}} @f$
        ///    - exp: return @f$ base_lr * gamma ^ iter @f$
        ///    - inv: return @f$ base_lr * {1 + gamma * iter} ^ {-power} @f$
        ///    - multistep: similar to step but it allows non-uniform steps defined by stepvalue.
        ///    - poly: the effective learning rate follows a polynomial decay, to be
        ///            zero by the max_iter.  return @f$ base_lr * {1 - iter/max_iter} ^ {power} @f$
        ///    - sigmoid: the effective learning rate follows a sigmoid decay.
        ///            return @f$ base_lr * {1/{1 + exp{-gamma * {iter - stepsize}}}} @f$
        ///            
        /// where base_lr, max_iter, gamma, step, stepvalue and power are defined int the
        /// solver protocol buffer, and iter is the current iteration.
        /// </remarks>
        [Category("Learning Policy")]
        [DisplayName("lr_policy")]
        [Description("Specifies the learning rate decay policy.  \n 'fixed' - always return base_lr.  \n 'step' - return base_lr * gamma ^ (floor(iter/step)).  \n 'exp' - return base_lr * gamma ^ iter.  \n 'inv' - return base_lr * (1 + gamma * iter) ^ (-power)." +
            "\n 'multistep' - similar to 'step' but allows non-uniform steps defined by stepvalue.  \n 'poly' - the effective learning rate follows a polynomial decay, to be zero by the max_iter, returns base_lr * (1 - iter/max_iter) ^ (power)." +
            "\n 'sigmoid' - the effective learning rate follows a sigmoid decay, returns base_lr * (1/(1 + exp(-gamma * (iter - stepsize)))).")]
        public LearningRatePolicyType LearningRatePolicy
        {
            get
            {
                switch (m_strLrPolicy)
                {
                    case "fixed":
                        return LearningRatePolicyType.FIXED;

                    case "step":
                        return LearningRatePolicyType.STEP;

                    case "exp":
                        return LearningRatePolicyType.EXP;

                    case "inv":
                        return LearningRatePolicyType.INV;

                    case "multistep":
                        return LearningRatePolicyType.MULTISTEP;

                    case "sigmoid":
                        return LearningRatePolicyType.SIGMOID;

                    case "poly":
                        return LearningRatePolicyType.POLY;

                    default:
                        throw new Exception("Unknown learning rate policy '" + m_strLrPolicy + "'");
                }
            }
            set
            {
                switch (value)
                {
                    case LearningRatePolicyType.FIXED:
                        m_strLrPolicy = "fixed";
                        break;

                    case LearningRatePolicyType.STEP:
                        m_strLrPolicy = "step";
                        break;

                    case LearningRatePolicyType.EXP:
                        m_strLrPolicy = "exp";
                        break;

                    case LearningRatePolicyType.INV:
                        m_strLrPolicy = "inv";
                        break;

                    case LearningRatePolicyType.MULTISTEP:
                        m_strLrPolicy = "multistep";
                        break;

                    case LearningRatePolicyType.SIGMOID:
                        m_strLrPolicy = "sigmoid";
                        break;

                    case LearningRatePolicyType.POLY:
                        m_strLrPolicy = "poly";
                        break;

                    default:
                        throw new Exception("Unknown learning rate policy '" + value.ToString() + "'.");
                }
            }
        }

        /// <summary>
        /// The learning rate decay policy.
        /// </summary>
        /// <remarks>
        /// The currently implemented learning rate policies are as follows:
        ///    - fixed: always return base_lr.
        ///    - step: return base_lr * gamma ^ (floor(iter / step))
        ///    - exp: return base_lr * gamma ^ iter
        ///    - inv: return base_lr * (1 + gamma * iter) ^ (-power)
        ///    - multistep: similar to step but it allows non-uniform steps defined by stepvalue.
        ///    - poly: the effective learning rate follows a polynomial decay, to be
        ///            zero by the max_iter.  return base_lr * (1 - iter/max_iter) ^ (power)
        ///    - sigmoid: the effective learning rate follows a sigmoid decay.
        ///            return base_lr * (1/(1 + exp(-gamma * (iter - stepsize))))
        ///            
        /// where base_lr, max_iter, gamma, step, stepvalue and power are defined int the
        /// solver protocol buffer, and iter is the current iteration.
        /// </remarks>
        [Browsable(false)]
        public string lr_policy
        {
            get { return m_strLrPolicy; }
            set { m_strLrPolicy = value; }
        }

        /// <summary>
        /// The 'gamma' parameter to compute the learning rate.
        /// </summary>
        [Category("Learning Policy")]
        [Description("Specifies the 'gamma' parameter to compute the 'step', 'exp', 'inv', and 'sigmoid' learning policy.")]
        public double gamma
        {
            get { return m_dfGamma; }
            set { m_dfGamma = value; }
        }

        /// <summary>
        /// The 'power' parameter to compute the learning rate.
        /// </summary>
        [Category("Learning Policy")]
        [Description("Specifies the 'power' parameter to compute the 'inv' and 'poly' learning policy.")]
        public double power
        {
            get { return m_dfPower; }
            set { m_dfPower = value; }
        }

        /// <summary>
        /// The momentum value.
        /// </summary>
        [Category("Solver - Not AdaGrad or RMSProp")]
        [Description("Specifies the momentum value - used by all solvers EXCEPT the 'AdaGrad' and 'RMSProp' solvers.")]
        public double momentum
        {
            get { return m_dfMomentum; }
            set { m_dfMomentum = value; }
        }

        /// <summary>
        /// The weight decay.
        /// </summary>
        [Description("Specifies the weight decay.")]
        public double weight_decay
        {
            get { return m_dfWeightDecay; }
            set { m_dfWeightDecay = value; }
        }

        /// <summary>
        /// The regularization type.
        /// </summary>
        /// <remarks>
        /// The regularization types supported are:
        ///    - L1 and L2 controlled by weight_decay.
        /// </remarks>
        [DisplayName("regularization_type")]
        [Description("Specifies the regularization type.  The regulation types supported are 'L1' and 'L2' controlled by weight decay.")]
        public RegularizationType Regularization
        {
            get
            {
                switch (m_strRegularizationType)
                {
                    case "L1":
                        return RegularizationType.L1;

                    case "L2":
                        return RegularizationType.L2;

                    default:
                        throw new Exception("Unknown regularization type '" + m_strRegularizationType + "'");
                }
            }
            set
            {
                switch (value)
                {
                    case RegularizationType.L1:
                        m_strRegularizationType = "L1";
                        break;

                    case RegularizationType.L2:
                        m_strRegularizationType = "L2";
                        break;

                    default:
                        throw new Exception("Unknown regularization type '" + value.ToString() + "'");
                }
            }
        }

        /// <summary>
        /// The regularization type.
        /// </summary>
        /// <remarks>
        /// The regularization types supported are:
        ///    - L1 and L2 controlled by weight_decay.
        /// </remarks>
        [Description("Specifies the regularization type.  The regulation types supported are 'L1' and 'L2' controlled by weight decay.")]
        [Browsable(false)]
        public string regularization_type
        {
            get { return m_strRegularizationType; }
            set { m_strRegularizationType = value; }
        }

        /// <summary>
        /// The stepsize for learning rate policy 'step'.
        /// </summary>
        [Category("Learning Policy")]
        [Description("Specifies the stepsize for the learning rate policy 'step'.")]
        public int stepsize
        {
            get { return m_nStepSize; }
            set { m_nStepSize = value; }
        }

        /// <summary>
        /// The step values for learning rate policy 'multistep'.
        /// </summary>
        [Category("Learning Policy")]
        [Description("Specifies the step values for the learning rate policy 'multistep'.")]
        public List<int> stepvalue
        {
            get { return m_rgnStepValue; }
            set { m_rgnStepValue = value; }
        }

        /// <summary>
        /// Set clip_gradients to >= 0 to clip parameter gradients to that L2 norm,
        /// whenever their actual L2 norm is larger.
        /// </summary>
        [Description("Set 'clip_gradients' to >= 0 to clip parameter gradients to that L2 norm, whenever their actual LT norm is larger.")]
        public double clip_gradients
        {
            get { return m_dfClipGradients; }
            set { m_dfClipGradients = value; }
        }

        /// <summary>
        /// The snapshot interval.
        /// </summary>
        [Category("Snapshot")]
        [Description("Sepcifies the snapshot interval.")]
        public int snapshot
        {
            get { return m_nSnapshot; }
            set { m_nSnapshot = value; }
        }

        /// <summary>
        /// The prefix for the snapshot.
        /// </summary>
        [Description("Specifies the snapshot prefix.")]
        [Browsable(false)]
        public string snapshot_prefix
        {
            get { return m_strSnapshotPrefix; }
            set { m_strSnapshotPrefix = value; }
        }

        /// <summary>
        /// Whether to snapshot diff in the results or not.  Snapshotting diff will help
        /// debugging but the final protocol buffer size will be much larger.
        /// </summary>
        [Category("Snapshot")]
        [Description("Specifies whether ot snapshot diff in the results or not.  Snapshotting diff may help debugging but the final snapshot data size will be much larger.")]
        public bool snapshot_diff
        {
            get { return m_bSnapshotDiff; }
            set { m_bSnapshotDiff = value; }
        }

        /// <summary>
        /// The snapshot format.
        /// </summary>
        /// <remarks>
        /// Currently only the Binary Proto Buffer format is supported.
        /// </remarks>
        [Description("Specifies the snapshot format.")]
        [Browsable(false)]
        public SnapshotFormat snapshot_format
        {
            get { return m_snapshotFormat; }
            set { m_snapshotFormat = value; }
        }

        /// <summary>
        /// Specifies whether or not the snapshot includes the trained weights.  The default = <i>true</i>.
        /// </summary>
        [Category("Snapshot")]
        [Description("Specifies whether or not the snapshot includes the trained weights.  The default = 'true'.")]
        public bool snapshot_include_weights
        {
            get { return m_bSnapshotIncludeWeights; }
            set { m_bSnapshotIncludeWeights = value; }
        }

        /// <summary>
        /// Specifies whether or not the snapshot includes the solver state.  The default = <i>false</i>.  Including the solver state will slow down the time of each snapshot.
        /// </summary>
        [Category("Snapshot")]
        [Description("Specifies whether or not the snapshot includes the solver state.  The default = 'false'.  Including the solver state will slow down the time of each snapshot.")]
        public bool snapshot_include_state
        {
            get { return m_bSnapshotIncludeState; }
            set { m_bSnapshotIncludeState = value; }
        }

        /// <summary>
        /// The device id that will be used when run on the GPU.
        /// </summary>
        [Description("Specifies the device ID that will be used when run on the GPU.")]
        [Browsable(false)]
        public int device_id
        {
            get { return m_nDeviceID; }
            set { m_nDeviceID = value; }
        }

        /// <summary>
        /// If non-negative, the seed with which the Solver will initialize the caffe
        /// random number generator -- useful for repoducible results.  Otherwise
        /// (and by default) initialize using a seed derived from the system clock.
        /// </summary>
        [Description("If non-negative, the seed with which the Solver will initialize the caffe random number generator -- useful for reproducible results.  Otherwise (and by default), initialize using a seed derived from the system clock.")]
        public long random_seed
        {
            get { return m_lRandomSeed; }
            set { m_lRandomSeed = value; }
        }

        /// <summary>
        /// Specifies the solver type.
        /// </summary>
        [Category("Solver")]
        [Description("Specifies the solver type. \n" +
                     "  SGD - stochastic gradient descent with momentum updates weights by a linear combination of the negative gradient and the previous weight update. \n" +
                     "  NESTEROV - Nesterov's accelerated gradient, similar to SGD, but error gradient is computed on the weights with added momentum. \n" +
                     "  ADADELTA - Gradient based optimization like SGD, see M. Zeiler 'Adadelta, An adaptive learning rate method', arXiv preprint, 2012. \n" +
                     "  ADAGRAD - Gradient based optimization like SGD that tries to find rarely seen features, see Duchi, E, and Y. Singer, 'Adaptive subgradient methods for online learning and stochastic optimization', The Journal of Machine Learning Research, 2011. \n" +
                     "  ADAM - Gradient based optimization like SGD that includes 'adaptive momentum estimation' and can be thougth of as a generalization of AdaGrad, see D. Kingma, J. Ba, 'Adam: A method for stochastic optimization', Intl' Conference for Learning Representations, 2015. \n" +
                     "  RMSPROP - Gradient based optimization like SGD, see T. Tieleman, and G. Hinton, 'RMSProp: Divide the gradient by a runnign average of its recent magnitude', COURSERA: Neural Networks for Machine Learning. Technical Report, 2012.")]
        public SolverType type
        {
            get { return m_solverType; }
            set { m_solverType = value; }
        }

        /// <summary>
        /// Numerical stability for RMSProp, AdaGrad, AdaDelta and Adam
        /// </summary>
        [Category("Solver - Ada and RMSProp")]
        [Description("Specifies the numerical stability for 'RMSProp', 'AdaGrad', 'AdaDelta' and 'Adam' solvers.")]
        public double delta
        {
            get { return m_dfDelta; }
            set { m_dfDelta = value; }
        }

        /// <summary>
        /// An additional momentum property for the Adam solver.
        /// </summary>
        [Category("Solver - Adam")]
        [Description("Specifies an additional momentum property used by the 'Adam' solver.")]
        public double momentum2
        {
            get { return m_dfMomentum2; }
            set { m_dfMomentum2 = value; }
        }

        /// <summary>
        /// RMSProp decay value.
        /// </summary>
        /// <remarks>
        /// MeanSquare(t) = rms_decay * MeanSquare(t-1) + (1 - rms_decay) * SquareGradient(t)
        /// </remarks>
        [Category("Solver - RMSProp")]
        [Description("Specifies the 'RMSProp' decay value used by the 'RMSProp' solver.  MeanSquare(t) = 'rms_decay' * MeanSquare(t-1) + (1 - 'rms_decay') * SquareGradient(t).  The 'momentum2' is only used by the 'RMSProp' solver.")]
        public double rms_decay
        {
            get { return m_dfRmsDecay; }
            set { m_dfRmsDecay = value; }
        }

        /// <summary>
        /// If true, print information about the state of the net that may help with
        /// debugging learning problems.
        /// </summary>
        [Category("Debug")]
        [Description("If true, print information about the sate of the net that may help with debugging learning problems.")]
        public bool debug_info
        {
            get { return m_bDebugInfo; }
            set { m_bDebugInfo = value; }
        }

        /// <summary>
        /// If false, don't save a snapshot after training finishes.
        /// </summary>
        [Category("Snapshot")]
        [Description("If false, don't save a snapshot after training finishes.")]
        public bool snapshot_after_train
        {
            get { return m_bSnapshotAfterTrain; }
            set { m_bSnapshotAfterTrain = value; }
        }

        /// <summary>
        /// Converts the SolverParameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name given to the RawProto.</param>
        /// <returns>The new RawProto representing the SolverParameter is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (net_param != null)
                rgChildren.Add(net_param.ToProto("net_param"));

            if (train_net_param != null)
                rgChildren.Add(train_net_param.ToProto("train_net_param"));

            foreach (NetParameter np in test_net_param)
            {
                rgChildren.Add(np.ToProto("test_net_param"));
            }

            if (train_state != null)
                rgChildren.Add(train_state.ToProto("train_state"));

            foreach (NetState ns in test_state)
            {
                rgChildren.Add(ns.ToProto("test_state"));
            }

            rgChildren.Add<int>("test_iter", test_iter);
            rgChildren.Add("test_interval", test_interval.ToString());
            rgChildren.Add("test_compute_loss", test_compute_loss.ToString());
            rgChildren.Add("test_initialization", test_initialization.ToString());
            rgChildren.Add("base_lr", base_lr.ToString());
            rgChildren.Add("display", display.ToString());
            rgChildren.Add("average_loss", average_loss.ToString());
            rgChildren.Add("max_iter", max_iter.ToString());

            if (iter_size != 1)
                rgChildren.Add("iter_size", iter_size.ToString());

            rgChildren.Add("lr_policy", lr_policy);

            if (lr_policy == "step" || lr_policy == "exp" || lr_policy == "inv" || lr_policy == "sigmoid")
                rgChildren.Add("gamma", gamma.ToString());

            if (lr_policy == "inv" || lr_policy == "poly")
                rgChildren.Add("power", power.ToString());

            rgChildren.Add("momentum", momentum.ToString());
            rgChildren.Add("weight_decay", weight_decay.ToString());
            rgChildren.Add("regularization_type", regularization_type);

            if (lr_policy == "step")
                rgChildren.Add("stepsize", stepsize.ToString());

            if (lr_policy == "multistep")
                rgChildren.Add<int>("stepvalue", stepvalue);

            if (clip_gradients >= 0)
                rgChildren.Add("clip_gradients", clip_gradients.ToString());

            rgChildren.Add("snapshot", snapshot.ToString());

            if (snapshot_prefix.Length > 0)
                rgChildren.Add("snapshot_prefix", snapshot_prefix);

            if (snapshot_diff != false)
                rgChildren.Add("snapshot_diff", snapshot_diff.ToString());

            rgChildren.Add("snapshot_format", snapshot_format.ToString());
            rgChildren.Add("device_id", device_id.ToString());
            
            if (random_seed >= 0)
                rgChildren.Add("ransom_seed", random_seed.ToString());

            rgChildren.Add("type", type.ToString());

            if (type == SolverType.RMSPROP || type == SolverType.ADAGRAD || type == SolverType.ADADELTA  || type == SolverType.ADAM)
                rgChildren.Add("delta", delta.ToString());

            if (type == SolverType.ADAM)
                rgChildren.Add("momentum2", momentum2.ToString());

            if (type == SolverType.RMSPROP)
                rgChildren.Add("rms_decay", rms_decay.ToString());

            if (debug_info != false)
                rgChildren.Add("debug_info", debug_info.ToString());
   
            if (snapshot_after_train != true)
                rgChildren.Add("snapshot_after_train", snapshot_after_train.ToString());

            if (!string.IsNullOrEmpty(custom_trainer))
                rgChildren.Add("custom_trainer", custom_trainer);

            if (!string.IsNullOrEmpty(custom_trainer_properties))
                rgChildren.Add("custom_trainer_properties", custom_trainer_properties);

            if (output_average_results != false)
                rgChildren.Add("output_average_results", output_average_results.ToString());

            rgChildren.Add("snapshot_include_weights", snapshot_include_weights.ToString());
            rgChildren.Add("snapshot_include_state", snapshot_include_state.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses a new SolverParameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto representing the SolverParameter.</param>
        /// <returns>The new SolverParameter instance is returned.</returns>
        public static SolverParameter FromProto(RawProto rp)
        {
            string strVal;
            SolverParameter p = new SolverParameter();

            RawProto rpNetParam = rp.FindChild("net_param");
            if (rpNetParam != null)
                p.net_param = NetParameter.FromProto(rpNetParam);

            RawProto rpTrainNetParam = rp.FindChild("train_net_param");
            if (rpTrainNetParam != null)
                p.train_net_param = NetParameter.FromProto(rpTrainNetParam);

            RawProtoCollection rgpTn = rp.FindChildren("test_net_param");
            foreach (RawProto rpTest in rgpTn)
            {
                p.test_net_param.Add(NetParameter.FromProto(rpTest));
            }

            RawProto rpTrainState = rp.FindChild("train_state");
            if (rpTrainState != null)
                p.train_state = NetState.FromProto(rpTrainState);

            RawProtoCollection rgpNs = rp.FindChildren("test_state");
            foreach (RawProto rpNs in rgpNs)
            {
                p.test_state.Add(NetState.FromProto(rpNs));
            }

            p.test_iter = rp.FindArray<int>("test_iter");

            if ((strVal = rp.FindValue("test_interval")) != null)
                p.test_interval = int.Parse(strVal);

            if ((strVal = rp.FindValue("test_compute_loss")) != null)
                p.test_compute_loss = bool.Parse(strVal);

            if ((strVal = rp.FindValue("test_initialization")) != null)
                p.test_initialization = bool.Parse(strVal);

            if ((strVal = rp.FindValue("base_lr")) != null)
                p.base_lr = double.Parse(strVal);

            if ((strVal = rp.FindValue("display")) != null)
                p.display = int.Parse(strVal);

            if ((strVal = rp.FindValue("average_loss")) != null)
                p.average_loss = int.Parse(strVal);

            if ((strVal = rp.FindValue("max_iter")) != null)
                p.max_iter = int.Parse(strVal);

            if ((strVal = rp.FindValue("iter_size")) != null)
                p.iter_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("lr_policy")) != null)
                p.lr_policy = strVal;

            if ((strVal = rp.FindValue("gamma")) != null)
                p.gamma = double.Parse(strVal);

            if ((strVal = rp.FindValue("power")) != null)
                p.power = double.Parse(strVal);

            if ((strVal = rp.FindValue("momentum")) != null)
                p.momentum = double.Parse(strVal);

            if ((strVal = rp.FindValue("weight_decay")) != null)
                p.weight_decay = double.Parse(strVal);

            if ((strVal = rp.FindValue("regularization_type")) != null)
                p.regularization_type = strVal;

            if ((strVal = rp.FindValue("stepsize")) != null)
                p.stepsize = int.Parse(strVal);

            p.stepvalue = rp.FindArray<int>("stepvalue");

            if ((strVal = rp.FindValue("clip_gradients")) != null)
                p.clip_gradients = double.Parse(strVal);

            if ((strVal = rp.FindValue("snapshot")) != null)
                p.snapshot = int.Parse(strVal);

            if ((strVal = rp.FindValue("snapshot_prefix")) != null)
                p.snapshot_prefix = strVal;

            if ((strVal = rp.FindValue("snapshot_diff")) != null)
                p.snapshot_diff = bool.Parse(strVal);

            if ((strVal = rp.FindValue("snapshot_format")) != null)
            {
                switch (strVal)
                {
                    case "BINARYPROTO":
                        p.snapshot_format = SnapshotFormat.BINARYPROTO;
                        break;

                    case "HDF5":
                        p.snapshot_format = SnapshotFormat.BINARYPROTO;                        
                        break;

                    default:
                        throw new Exception("Unknown 'snapshot_format' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("device_id")) != null)
                p.device_id = int.Parse(strVal);

            if ((strVal = rp.FindValue("random_seed")) != null)
                p.random_seed = long.Parse(strVal);

            if ((strVal = rp.FindValue("type")) != null)
            {
                string strVal1 = strVal.ToLower();

                switch (strVal1)
                {
                    case "sgd":
                        p.type = SolverType.SGD;
                        break;

                    case "nesterov":
                        p.type = SolverType.NESTEROV;
                        break;

                    case "adagrad":
                        p.type = SolverType.ADAGRAD;
                        break;

                    case "adadelta":
                        p.type = SolverType.ADADELTA;
                        break;

                    case "adam":
                        p.type = SolverType.ADAM;
                        break;

                    case "rmsprop":
                        p.type = SolverType.RMSPROP;
                        break;

                    default:
                        throw new Exception("Unknown solver 'type' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("delta")) != null)
                p.delta = double.Parse(strVal);

            if ((strVal = rp.FindValue("momentum2")) != null)
                p.momentum2 = double.Parse(strVal);

            if ((strVal = rp.FindValue("rms_decay")) != null)
                p.rms_decay = double.Parse(strVal);

            if ((strVal = rp.FindValue("debug_info")) != null)
                p.debug_info = bool.Parse(strVal);

            if ((strVal = rp.FindValue("snapshot_after_train")) != null)
                p.snapshot_after_train = bool.Parse(strVal);

            if ((strVal = rp.FindValue("custom_trainer")) != null)
                p.custom_trainer = strVal;

            if ((strVal = rp.FindValue("custom_trainer_properties")) != null)
                p.custom_trainer_properties = strVal;

            if ((strVal = rp.FindValue("output_average_results")) != null)
                p.output_average_results = bool.Parse(strVal);

            if ((strVal = rp.FindValue("snapshot_include_weights")) != null)
                p.snapshot_include_weights = bool.Parse(strVal);

            if ((strVal = rp.FindValue("snapshot_include_state")) != null)
                p.snapshot_include_state = bool.Parse(strVal);

            return p;
        }

        /// <summary>
        /// Returns a debug string for the SolverParameter.
        /// </summary>
        /// <returns>The debug string is returned.</returns>
        public string DebugString()
        {
            return m_solverType.ToString();
        }
    }
}
