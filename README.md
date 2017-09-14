<H2>Welcome to MyCaffe!</H2>

<b><a href="http://mycaffe.ai">MyCaffe</a></b> is a complete C# re-write of the <b>C</b>onvolutional <b>A</b>rchitecture for <b>F</b>ast <b>F</b>eature <b>E</b>ncoding (CAFFE[1]), 
the <a href="http://caffe.berkeleyvision.org/">open-source C++ Caffe</a> project, originally created by Yangquing Jia, and currently maintained by <a href="http://bair.berkeley.edu/">Berkeley AI Research</a>
with Evan Shelhamer as Lead Developer.

MyCaffe allows Windows C# software developers to use and expand deep learning solutions in their native C# language.  All layers except for a few, and nearly every unit test are now provided in C#.
Windows programmers can now write their own custom layers in the C# language, yet still enjoy the benefit of an efficient deep learning architecture that supports multi-GPU training on up to 8 GPU's
using <a href="https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/">NVIDIA's NCCL ('Nickel')</a>. 

Now you can create custom layers for MyCaffe in C# using the full extent of the <a href="https://msdn.microsoft.com/en-us/library/w0x726c2(v=vs.110).aspx">Windows .NET Framwork</a>!

We have made a large effort to keep the MyCaffe C# code true to the original down to comment with the hope of making it even easier to extend 
the general Caffe architecture for all.  In addition, MyCaffe uses the same Proto Buffer file format for solver and model descriptions and model 
binary files allowing an easy exchange between the MyCaffe and C++ Caffe platforms.  

Most of the MyCaffe C# code is very similar to the C++ Caffe code, for our goal is to extend the Caffe platform to C# programmers, while 
maintaining compatibility with Caffe's solver descriptions, model descriptions and binary weight format.

To get started with the MyCaffe C# source, check out the <a href="https://github.com/mycaffe">MyCaffe open-source</a> project on GitHub.  

The C# based MyCaffe open-source project is independently maintained by <a href="http://www.signalpop.com">SignalPOP LLC</a> and made 
available under the Apache 2.0 License.
<h3>Supported Development Environments:</h3>
* Visual Studio 2017 & <a href="https://developer.nvidia.com/cuda-toolkit/whatsnew">CUDA 9.0RC</a> & <a href="https://developer.nvidia.com/cudnn">cuDnn 7.0</a> (currently uses VS2015 compile)</br>
* Visual Studio 2017 & <a href="https://developer.nvidia.com/cuda-toolkit/whatsnew">CUDA 8.0</a> & <a href="https://developer.nvidia.com/cudnn">cuDnn 6.0</a> (currently uses VS2015 compile)</br>
* Visual Studio 2015 & <a href="https://developer.nvidia.com/cuda-toolkit/whatsnew">CUDA 9.0RC</a> & <a href="https://developer.nvidia.com/cudnn">cuDnn 7.0</a> </br>
* Visual Studio 2015 & <a href="https://developer.nvidia.com/cuda-toolkit/whatsnew">CUDA 8.0</a> & <a href="https://developer.nvidia.com/cudnn">cuDnn 6.0</a> </br>
</br>

IMPORTANT: The open-source MyCaffe project on GitHub is considered 'pre-release' and will have bugs.  When you find bugs or other issues, please report them here - or better yet, get involved
and propose a fix!

[1] [Caffe: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093) by Yangqing Jai, Evan Shelhamer, Jeff Donahue, 
Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell, 2014.

