<H2>Welcome to MyCaffe!</H2>

<b><a href="https://github.com/mycaffe">MyCaffe</a></b> is a complete C# re-write of the native C++ CAFFE[1] open source project.  

MyCaffe allows Windows C# software developers to use and expand deep learning solutions in their native C# language.  All layers except for a few, and nearly every unit test are now
provided in C#.  Windows programmers can now write their own custom layers in the C# langauge, yet still enjoy the benefit of an efficient deep learning architecture that supports 
multi-GPU training on up to 8 headless GPU's using <a href="https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/">NCCL 1.3.4 ('Nickel')</a>.

Now you can create custom layers for MyCaffe in C# using the full extent of the <a href="https://msdn.microsoft.com/en-us/library/w0x726c2(v=vs.110).aspx">Windows .NET Framwork</a>!

We have made a large effort to keep the MyCaffe C# code true to the original down to comment with the hope of making it even easier to extend the general CAFFE architecture for all.  
In addition, MyCaffe uses the same Proto Buffer file format for solver and model descriptions and model binary files allowing an easy exchange between the MyCaffe and C++ %CAFFE platforms.  

Most of the MyCaffe C# code is very similar to the C++ %CAFFE code, for our goal is to extend the %CAFFE platform to C# programmers, while 
maintaining compatibility with %CAFFE's solver descriptions, model descriptions and binary weight format.

To get started with the MyCaffe C# source, check out the <a href="https://github.com/mycaffe">MyCaffe open-source</a> project on GitHub.  

The C# based MyCaffe open-source project is independently maintained by <a href="http://www.signalpop.com">SignalPop LLC</a> and made available under the Appache 2.0 License.

[1] [Caffe: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093) by Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell, 2014.

For more information on the C++ CAFFE open-source project, please see the following <a href="http://caffe.berkeleyvision.org/">link</a>.