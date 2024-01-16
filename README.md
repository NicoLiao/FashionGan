
<p align=center>
<img target = "banner" src="messageImage_1704986745514.jpg">
</p>

<p align=center>
<a target="badge" href="https://github.com/NicoLiao/Pixiv-Downloader" title="python version"><img src="https://img.shields.io/badge/python-v3.10.12-blue"></a>
<a target="badge" href="https://github.com/NicoLiao/Pixiv-Downloader" title="windows badge"><img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" width=85/></a>  
</p>

>This is a software to generate fashion clothes base on the dataset.

All data comes from https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?resource=download

# Install

## Download source code
In order to use FashionGAN, make sure that you have python 3.10.12 and python packages as below:

* pytorch 2.0.1+cu117
* numpy 1.24.1
* matplotlib 3.7.2

```
$ git clone https://github.com/NicoLiao/FashionGan
```
## Usage
### Visual Studio Code
Download VSCode https://code.visualstudio.com/

Download python https://www.python.org/

After installing VScode and Python, download the extension in VSCode as follow:
* python
* python for vscode
* code runner

Done it and Run code

### Build exe
Using terminal and pyinstaller to build exe
```
$ pyinstaller -F --noconsole GUI.py 
```
