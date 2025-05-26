# '''
# Example of using DataMax for parsing: Currently supported file types include: doc, docx, epub, ppt, pptx, html, pdf, txt, xls, xlsx, jpg(png|jpeg|...)
# '''
#
# # 1. Import the SDK parsing class
from datamax import DataMax
import time

# 2. Use the corresponding parsing class to parse the file
data = DataMax(file_path=r"C:\Users\cykro\Desktop\纯文本船视宝.txt")
print(f'txt parser result:-->{data.get_data()}')

# "XLSX files may encounter long-running calculations (e.g., complex formulas or large datasets). You can set a TTL (Time-To-Live) to enforce a timeout."
data = DataMax(file_path=r"db.xlsx", timeout=10)
print(f'xlsx parser result:-->{data.get_data()}')



#
# Data Cleaning
"""
For specific cleaning rules, please refer to datamax/utils/data_cleaner.py
abnormal: abnormal data cleaning
private: privacy processing
filter: text filtering
"""
# Direct use: Supports direct cleaning of text content in the text parameter and returns a string
dm = DataMax()
data1 = dm.clean_data(method_list=["abnormal", "private"], text="<div></div>你好 18717777777 \n\n\n\n")
print(data1)

cleaned_data = dm.clean_data(method_list=["abnormal", "filter", "private"])
print(cleaned_data)


# Data prelabeling (new)
dm = DataMax(file_path=r"C:\Users\cykro\Desktop\项目交接文档.md")
data = dm.get_pre_label(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    model_name="qwen-max",
    chunk_size=500,
    chunk_overlap=100,
    question_number=5,
    max_workers=5,
    # message=[]
)
print(data)