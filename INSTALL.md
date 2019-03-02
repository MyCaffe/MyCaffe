<H2>Installation Instructions</H2>
To install and run <b>MyCaffe</b> you will need to do the following steps.  As a side note, we are using (and recommend) CUDA 10.1.105 with cuDNN 7.5.0 and Visual Studio 2017 on Windows 10 Pro for all of our testing.
</br>
<H3>I. CUDA - Install NVIDIA CUDA and cuDNN Libraries</H3>
Install CUDA 10.1 as shown below.
<H4>A. CUDA 10.1 - Install NVIDIA CUDA and cuDNN Libraries</H4>
1.) Install the NVIDIA CUDA 10.1.105 Toolkit for Windows 10 from https://developer.nvidia.com/cuda-downloads. 
</br>2.) Install the NVIDIA cuDNN 7.5.0 Accelerated Libraries for CUDA 10.1 on Windows 10 from https://developer.nvidia.com/cuDNN.
</br>3.) Create a new directory off your <b><i>$(CUDA_PATH_V10_1)</i></b> installation location named <b><i>cudann_10.1-win-v7.5.0.56</i></b> and copy the cuDNN <b><i>cudnn.h</i></b> and <b><i>cudnn.lib</i></b> files into it.
</br>4.) Copy the <b><i>cudnn64_7.dll</i></b> file into the <b><i>$(CUDA_PATH_V10_1)\bin</i></b> directory.
</br>5.) Install the NVIDIA NVAPI (r410) from https://developer.nvidia.com/nvapi.
</br>6.) Create anew directory off your <b><i>$(CUDA_PATH_V10_1)</i></b> installation location named <b><i>nvapi_410</i></b> and copy the NVAPI header and library files into it.
</br>
</br>NOTE: The CudaDnnDLL project points to the file directories noted above for the cuDNN include and library files.  

<H3>II. Setup Strong Names and Signing</H3>
The <b><i>MyCaffe</i></b> project, uses the following strong name key files:
</br>The <b>CudaControl</b> uses the <b><i>CudaControl.pfx</i></b> located in the <b><i>packages\CudaControl.0.10.1.x\lib\Net40\</i></b> directory.  
If you download, build the <b>CudaControl</b> repository and create a new <b><i>CudaControl.pfx</i></b> file, you should also copy it into the 
<b><i>packages\CudaControl.0.10.1.x\lib\Net40\</i></b> directory, replacing the pfx file there.  Alternatively, you can just install 
the <b>CudaControl</b> package from NuGet.
</p>
The <b><i>MyCaffe</i></b> uses the <b><i>mycaffe.sn.pfx</i></b> key file for string name signing.
</br>The <b><i>MyCaffe.basecode</i></b> uses the <b><i>mycaffe.basecode.sn.pfx</i></b> key file for string name signing.
</br>The <b><i>MyCaffe.imagedb</i></b> uses the <b><i>mycaffe.imagedb.sn.pfx</i></b> key file for string name signing.
</p>
You may want to provide your own strong names for each of the other <b>MyCaffe</b> projects.  To do this just select the project <i>Properties | Signing</i> tab and
then <i>Sign the assembly</i> with a strong name keyfile.  You can also use this method to create the <b><i>CudaControl.pfx</i></b> file mentioned above.
</br>If you change these, please do not check them in.  NOTE: The current .gitignore file ignores pfx files.

<H3>III. Required Software</H3>
<b>MyCaffe</b> requires the following software.
</br>
</br>a.) Microsoft Visual Studio 2017 (recommended) or Visual Studio 2015
</br>b.) Microsoft SQL or Microsoft SQL Express
</br>Both 'a' and 'b' are available from Microsoft at www.microsoft.com.
</br>
</br>c.) nccl64_134.10.1.dll - If you plan on running multi-GPU training sessions, you will need the <b><i>nccl64_134.10.1.dll</i></b>, which must be placed
in a directory that is visible by your executable files.  This library can be built from the MyCaffe\NCCL repository.  Alternatively, it is installed
by the <b>CudaControl</b> NuGet package and placed in the <i>packages\CudaControl.0.10.1.x\lib\Net40</i> directory.  You should copy the library into
a directory that is visible by your executable files.  NOTE: The automated multi-GPU tests use GPU's 1-4 where the monitor is plugged into GPU 0.
</br>
<H3>V. Create The Database</H3>
<b>MyCaffe</b> uses Microsoft SQL (or Microsoft SQL Express) as its underlying database.  You will need to create the database with the following steps:
</br>1.) Run the <b>MyCaffe.app.exe</b> test application and select the '<i>Database | Create Database</i>' menu item.
</br>2.) From the <i>Create Database</i> dialog, selecd the location where you want to create the database and select the <i>OK</i> button.
</br>3.) Next, load the <b>MNIST</b> data by selecting the '<i>Database | Load MNIST...'</i> menu item and follow the steps to get the data files.
</br>4.) Next, load the <b>CIFAR-10</b> data by selecting the '<i>Database | Load CIFAR-10...'</i> menu item and follow the steps to get the data files.
</br>NOTE: The automated tests expect that both the MNIST and CIFAR-10 datasets have already been loaded into the database.
<H2>Test Installation Instructions</H2>
To install data used by the Automated Tests, you will need to install the following files:
</br>
</br>See <a href=".\MyCaffe.test\test_data\models\bvlc_nin\INSTALL.md">Installing BVLC NIN Model</a>
</br>See <a href=".\MyCaffe.test\test_data\models\voc_fcns32\INSTALL.md">Installing BLVC_FCN Model</a>
</br>
</br>Both of these models are used by the <b><i>TestPersistCaffe.cs</i></b> auto tests.

