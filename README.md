# DataMax(CPU without PDF)

## Overview
DataMax is designed as a comprehensive solution for processing diverse file formats, performing data cleaning, and facilitating data annotation.

## Key Features

### File Processing Capabilities
Currently supports reading, conversion, and extraction from:
- HTML  
- DOCX/DOC, PPT/PPTX  
- EPUB
- XLS/XLSX spreadsheets  
- Plain text (TXT)  

### Data Cleaning Pipeline
Three-tiered cleaning process:
1. Anomaly detection and handling  
2. Privacy protection processing  
3. Text filtering and normalization  

### AI-Powered Data Annotation
Implements an LLM+Prompt to:
- Continuously generate pre-labeled datasets  
- Provide optimized training data for model fine-tuning  


## Installation Guide (Key Dependencies)
Dependencies include libreoffice, datamax, and MinerU.

### 1. Installing libreoffice Dependency
**Note:** Without datamax, .doc files will not be supported.

#### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install libreoffice
```
### Windows
```text
Install LibreOffice from: [Download LibreOffice](https://www.libreoffice.org/download/download-libreoffice/?spm=5176.28103460.0.0.5b295d275bpHzh)  
Add to environment variable: `$env:PATH += ";C:\Program Files\LibreOffice\program"`
```
### Checking LibreOffice Installation
```bash
soffice --version
```



##  2. Installing Basic Dependencies for datamax
1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies into conda:
   ```bash
   cd datamax
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```


## Features
- **Multi-format Support**: Capable of handling various text file types such as PDF, HTML, DOCX, and TXT.
- **Content Extraction**: Provides powerful content extraction capabilities to accurately retrieve information from complex document structures.
- **Data Conversion**: Supports converting processed data into markdown format for further analysis.
- **Batch Processing**: Can handle multiple files at once, improving work efficiency.
- **Customizable Configuration**: Users can adjust processing parameters according to their needs to meet different business requirements.
- **Cross-platform Compatibility**: This SDK can run on multiple operating systems, including Windows, MacOS, and Linux.


## Technology Stack

- **Programming Language**: Python >= 3.10  
- **Dependency Libraries**:
  - BeautifulSoup: For HTML file parsing.  
  - python-docx: For DOCX file parsing.  
  - pandas: For data processing and conversion.  

- **Development Environment**: Visual Studio Code or PyCharm  
- **Version Control**: Git  

## Usage Instructions
### Installing the SDK
- **Installation Commands**:
  ```bash
  ## Local Installation
  python setup.py sdist bdist_wheel
  pip install dist/datamax-0.1.3-py3-none-any.whl
  
  ## Pip Installation
  pip install pydatamax
  ```
  

- **Importing the Code**:
    ```python
    # File Parsing
    from datamax import DataMax
    
    ## Handling a Single File in Two Ways
    # 1. Using a List of Length 1
    data = DataMax(file_path=[r"docx_files_example/船视宝概述.doc"])
    data = data.get_data()
    
    # 2. Using a String
    data = DataMax(file_path=r"docx_files_example/船视宝概述.doc")
    data = data.get_data()
    
    ## Handling Multiple Files
    # 1. Using a List of Length n
    data = DataMax(file_path=[r"docx_files_example/船视宝概述1.doc", r"docx_files_example/船视宝概述2.doc"])
    data = data.get_data()
    
    # 2. Passing a Folder Path as a String
    data = DataMax(file_path=r"docx_files_example/")
    data = data.get_data()
    
    # Data Cleaning
    """
    Cleaning rules can be found in datamax/utils/data_cleaner.py
    abnormal: Abnormal cleaning
    private: Privacy processing
    filter: Text filtering
    """
    # Direct Use: Clean the text parameter directly and return a string
    dm = DataMax()
    data = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
    
    # supported after use `get_data()`
    dm = DataMax(file_path=r"C:\Users\cykro\Desktop\数据库开发手册.docx")
    data2 = dm.get_data()
    cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
    
    # Large Model Pre-annotation Supporting any model that can be called via OpenAI SDK
    data = DataMax(file_path=r"path\to\xxx.docx")
    parsed_data = data.get_data()
    # If no custom messages are passed, the default messages in the SDK will be used
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'}
    ]
    qa_datas = data.get_pre_label(
        api_key="sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_name="qwen-max",
        chunk_size=500,
        chunk_overlap=100,
        question_number=5,
        max_workers=5,
        # message=[]
    )
    print(f'Annotated result:{qa_datas}')
    ```


## Examples
    ```python
    ## docx | doc | epub | html | txt | ppt | pptx | xls | xlsx
    from datamax import DataMax
    data = DataMax(file_path=r"docx_files_example/船视宝概述.doc", to_markdown=True)
    """
    Parameters: 
    file_path: Relative file path / Absolute file path
    to_markdown: Whether to convert to markdown (default value False, directly returns text) This parameter only supports word files (doc | docx)
    """
    ```

## Contribution Guide
We welcome any form of contribution, whether it is reporting bugs, suggesting new features, or submitting code improvements. Please read our Contributor's Guide to learn how to get started.
## License
This project is licensed under the MIT License. For more details, see the LICENSE file.

## Contact Information
If you encounter any issues during use, or have any suggestions or feedback, please contact us through the following means:
- Email: cy.kron@foxmail.com
- Project Homepage: GitHub Project Link

