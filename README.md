<H2>Welcome to MyCaffe!</H2>

<b><a href="https://github.com/mycaffe">MyCaffe</a></b> is a complete C# re-write of the native C++ CAFFE[1] open source project.

MyCaffe allows Windows C# software developers to use and expand deep learning solutions in their native C# language.  All layers except for a few, and nearly every unit test are now provided in C#.
Windows programmers can now write their own custom layers in the C# language, yet still enjoy the benefit of an efficient deep learning architecture that supports multi-GPU training on up to 8 
headless GPU's using <a href="https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/">NCCL 1.3.4 ('Nickel')</a>. 

Now you can create custom layers for MyCaffe in native C# using the full extent of the <a href="https://msdn.microsoft.com/en-us/library/w0x726c2(v=vs.110).aspx">Windows .NET Framwork</a>!

We have made a large effort to keep the MyCaffe C# code true to the original down to comment with the hope of making it even easier to extend 
the general CAFFE architecture for all.  In addition, MyCaffe uses the same Proto Buffer file format for solver and model descriptions and model 
binary files allowing an easy exchange between the MyCaffe and C++ CAFFE platforms.  

Most of the MyCaffe C# code is very similar to the C++ CAFFE code, for our goal is to extend the CAFFE platform to C# programmers, while 
maintaining compatibility with CAFFE's solver descriptions, model descriptions and binary weight format.

The C# based MyCaffe open-source project is independently maintained by <a href="http://www.signalpop.com">SignalPop LLC</a> and made 
available under the Apache 2.0 License.
<h3>Supported Development Environments:</h3>
* Visual Studio 2017 & <a href="https://developer.nvidia.com/cuda-toolkit/whatsnew">CUDA 9.1</a> & <a href="https://developer.nvidia.com/cudnn">cuDnn 7.1.3</a> (currently uses VS2015 compile)</br>
* Visual Studio 2015 & <a href="https://developer.nvidia.com/cuda-toolkit/whatsnew">CUDA 9.1</a> & <a href="https://developer.nvidia.com/cudnn">cuDnn 7.1.3</a> </br>
</br>

NOTE: Visual Studio 2017 support noted above currently uses the Platform Toolset "Visual Studio 2015 (v140)" build setting.

IMPORTANT: The open-source MyCaffe project on GitHub is considered 'pre-release' and will have bugs.  When you find bugs or other issues, please report them here - or better yet, get involved
and propose a fix!

MyCaffe now supports source and target datasets and the gradient reversal layer which allow implementing Domain-Adversarial Neural Networks (DANN) as described in [2].

[1] [Caffe: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093) by Yangqing Jai, Evan Shelhamer, Jeff Donahue, 
Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell, 2014.

[2] [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) by Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, 
Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky, 2015.

For more information on the C++ CAFFE open-source project, please see the following <a href="http://caffe.berkeleyvision.org/">link</a>.

