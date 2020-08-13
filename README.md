<H2>Welcome to MyCaffe!</H2>

<b><a href="https://github.com/mycaffe">MyCaffe</a></b> is a complete C# re-write of the native C++ CAFFE[1] open source project.

MyCaffe allows Windows C# software developers to use and expand deep learning solutions in their native C# language.  All layers except for a few, and nearly every unit test are now provided in C#.
Windows programmers can now write their own custom layers in the C# language, yet still enjoy the benefit of an efficient deep learning architecture that supports multi-GPU training on up to 8 
headless GPU's using <a href="https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/">NCCL 1.3.4 ('Nickel')</a>. 

Now you can create custom layers for MyCaffe in native C# using the full extent of the <a href="https://msdn.microsoft.com/en-us/library/w0x726c2(v=vs.110).aspx">Windows .NET Framwork</a>!

We have made a large effort to keep the MyCaffe C# code true to the original CAFFE[1] down to comment with the hope of making it even easier to extend 
the general CAFFE architecture for all.  In addition, MyCaffe uses the same Proto Buffer file format for solver and model descriptions and model 
binary files allowing an easy exchange between the MyCaffe and C++ CAFFE platforms.  

Most of the MyCaffe C# code is very similar to the C++ CAFFE code, for our goal is to extend the CAFFE platform to C# programmers, while 
maintaining compatibility with CAFFE's solver descriptions, model descriptions and binary weight format.

The C# based MyCaffe open-source project is independently maintained by <a href="http://www.signalpop.com">SignalPop LLC</a> and made 
available under the Apache 2.0 License.
<h3>Supported Development Environments:</h3>
* Visual Studio 2017 & <a href="https://developer.nvidia.com/cuda-toolkit/whatsnew">CUDA 11.0.2</a> & <a href="https://developer.nvidia.com/cudnn">cuDnn 8.0.2</a> </br>
</br>

NOTE: Compute 5.3 required for CUDA 11.0.2/cuDNN 8.0.2 to support __half sized memory.

<b>For detailed notes on building MyCaffe, please see the <a href="https://github.com/MyCaffe/MyCaffe/blob/master/INSTALL.md">INSTALL.md</a> file.</b>

<b>IMPORTANT</b>: The open-source MyCaffe project on GitHub is considered 'pre-release' and may have bugs.  When you find bugs or other issues, please report them here - or better yet, get involved
and propose a fix!

We have several new models supported by MyCaffe with the train_val and solution prototxt ready to go:
 - Domain-Adversarial Neural Networks (DANN) as described in [2] with support for source and target datasets.
 - ResNet-56 on the Cifar-10 dataset as described in [3].
 - Deep convolutional auto-encoder neural networks with pooling as described in [4].
 - Policy Gradient Reinforcement Learning networks as described in [5].
 - Recurrent Learning of Char-RNN as described in [8] and [9].
 - Neural Style Transfer as described in [10] and [11] using the VGG model described in [12]
 - Deep Q-Learning [14][15] with Noisy-Net [16] and Prioritized Replay Buffer [17]
 - Siamese Network [18][19]
 - Deep Metric Learning with Triplet Network [20][21]

For more information on the MyCaffe implementation of Policy Gradient Reinforcement Learning, see [MyCaffe: A Complete C# Re-Write of Caffe with Reinforcement Learning](https://arxiv.org/abs/1810.02272) by D. Brown, 2018. 

MyCaffe now supports the Arcade-Learning-Environment by [6] based on the Stella Atari-2600 emulator from [7], via the AleControl from SignalPop.  
For more information, get the <a href="https://www.nuget.org/packages?q=AleControl">AleControl on Nuget</a>, or visit the <a href="https://github.com/MyCaffe/AleControl">AleControl on Github</a>.

<h2>License and Citation</h2>
MyCaffe is released under the [Apache License 2.0](https://github.com/MyCaffe/MyCaffe/blob/master/LICENSE).  

Please cite MyCaffe in your publications and projects if MyCaffe helps you in your research or applications:
<pre>
<code>
	@article 
	{
	  brown2018mycaffe,
	  Author = {Brown, David W.}
	  Journal = {arXiv preprint arXiv:1810.02272},
	  Title = {MyCaffe: A Complete C# Re-Write of Caffe with Reinforcement Learning}
	  Year = {2018}
	  Link = {https://arxiv.org/abs/1810.02272}
	}
</code>
</pre>

[1] [CAFFE: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093) by Yangqing Jai, Evan Shelhamer, Jeff Donahue, 
Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell, 2014.

[2] [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) by Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, 
Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky, 2015.

[3] [ResNet 20/32/44/56/110 for CIFAR10 with caffe](https://github.com/yihui-he/resnet-cifar10-caffe) by Yihui He, 2016.

[4] [A Deep Convolutional Auto-Encoder with Pooling - Unpooling Layers in Caffe](https://arxiv.org/abs/1701.04949) by Volodymyr Turchenko, Eric Chalmers and Artur Luczac, 2017.

[5] [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) by Andrej Karpathy, 2015.

[6] [The Arcade Learning Environment: An Evaluation Platform for General Agents](https://arxiv.org/abs/1207.4708) by Marc G. Bellemare, 
Yavar Naddaf, Joel Veness and Michael Bowling, 2012-2013.  Source code available on GitHub at <a href="https://github.com/mgbellemare/Arcade-Learning-Environment">mgbellemare/Arcade-Learning-Envrionment</a>

[7] [Stella - A multi-platform Atari 2600 VCS emulator](https://stella-emu.github.io/) by Bradford W. Mott, Stephen Anthony and The Stella Team, 1995-2018
Source code available on GitHub at <a href="https://github.com/stella-emu/stella">stella-emu/stella</a>

[8] [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy, 2015.

[9] [adepierre/caffe-char-rnn Github](https://github.com/adepierre/caffe-char-rnn) by adepierre, 2017.

[10] [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, 2015, arXiv:1508:06576

[11] [ftokarev/caffe Github](https://github.com/ftokarev/caffe-neural-style) by ftokarev, 2017

[12] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) by K. Simonyan, A. Zisserman, arXiv:1409.1556

[14] [GitHub: Google/dopamine](https://github.com/google/dopamine) licensed under the [Apache 2.0 License](https://github.com/google/dopamine/blob/master/LICENSE);

[15] [Dopamine: A Research Framework for Deep Reinforcement Learning](https://arxiv.org/abs/1812.06110) by Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, Marc G. Bellemare, 2018, arXiv:1812.06110

[16] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) by Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, Shane Legg, 2018, arXiv:1706.10295

[17] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) by Tom Schaul, John Quan, Ioannis Antonoglou, David Silver, 2016, arXiv:1511.05952

[18] [Siamese Network Training with Caffe](https://caffe.berkeleyvision.org/gathered/examples/siamese.html) by Yangqing Jia and Evan Shelhamer, BAIR.

[19] [Siamese Neural Network for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by G. Koch, R. Zemel and R. Salakhutdinov, ICML 2015 Deep Learning Workshop, 2015.

[20] [Deep metric learning using Triplet network](https://arxiv.org/abs/1412.6622) by E. Hoffer and N. Ailon, 2014, 2018, arXiv:1412.6622.

[21] [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by A. Hermans, L. Beyer, and B. Leibe, 2017, arXiv:1703.07737v2.

For more information on the C++ CAFFE open-source project, please see the following <a href="http://caffe.berkeleyvision.org/">link</a>.

