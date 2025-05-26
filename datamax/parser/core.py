import os
import importlib
from typing import List, Union, Dict
from datamax.utils import data_cleaner
from datamax.utils.qa_generator import generatr_qa_pairs


class ParserFactory:
    @staticmethod
    def create_parser(
            file_path: str,
            to_markdown: bool = False,
            timeout: int = 1200
    ):
        """
        Create a parser instance based on the file extension.
        :param file_path: The path to the file to be parsed.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
                    (only supported files in .doc or .docx format)
        :param timeout: Timeout for the request .(only supported files in .xlsx format)
        :return: An instance of the parser class corresponding to the file extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        parser_class_name = {
            '.md': 'MarkdownParser',
            '.docx': 'DocxParser',
            '.doc': 'DocParser',
            '.epub': 'EpubParser',
            '.html': 'HtmlParser',
            '.txt': 'TxtParser',
            '.pptx': 'PPtxParser',
            '.ppt': 'PPtParser',
            '.xlsx': 'XlsxParser',
            '.xls': 'XlsParser'
        }.get(file_extension)

        if not parser_class_name:
            return None

        # Dynamically determine the module name based on the file extension
        module_name = f'datamax.parser.{file_extension[1:]}_parser'

        try:
            # Dynamically import the module and get the class
            module = importlib.import_module(module_name)
            parser_class = getattr(module, parser_class_name)

            if parser_class_name == 'DocxParser' or parser_class_name == 'DocParser':
                return parser_class(
                    file_path=file_path,
                    to_markdown=to_markdown
                )
            elif parser_class_name == 'XlsxParser':
                return parser_class(
                    file_path=file_path,
                    timeout=timeout
                )
            else:
                return parser_class(
                    file_path=file_path
                )

        except (ImportError, AttributeError) as e:
            raise e


class DataMax:
    def __init__(self,
                 file_path: Union[str, list] = '',
                 to_markdown: bool = False,
                 timeout: int = 1200
                 ):
        """
        Initialize the DataMaxParser with file path and parsing options.
        :param file_path: The path to the file or directory to be parsed.
        :param to_markdown: Flag to indicate whether the output should be in Markdown format.
        """
        self.file_path = file_path
        self.to_markdown = to_markdown
        self.parsed_data = None
        self.timeout = timeout

    def get_data(self):
        """
        Parse the file or directory specified in the file path and return the data.

        :return: A list of parsed data if the file path is a directory, otherwise a single parsed data.
        """
        try:
            if isinstance(self.file_path, list):
                parsed_data = [self._parse_file(f) for f in self.file_path]
                self.parsed_data = parsed_data
                return parsed_data

            elif isinstance(self.file_path, str) and os.path.isfile(self.file_path):
                parsed_data = self._parse_file(self.file_path)
                self.parsed_data = parsed_data
                return parsed_data

            elif isinstance(self.file_path, str) and os.path.isdir(self.file_path):
                file_list = [os.path.join(self.file_path, file) for file in os.listdir(self.file_path)]
                parsed_data = [self._parse_file(f) for f in file_list if os.path.isfile(f)]
                self.parsed_data = parsed_data
                return parsed_data
            else:
                raise ValueError("Invalid file path.")

        except Exception as e:
            raise e

    def clean_data(self, method_list: List[str], text: str = None):
        """
        Clean data

        methods include AbnormalCleaner， TextFilter， PrivacyDesensitization which is 1 2 3

        :return:
        """
        if text:
            cleaned_text = text
        elif self.parsed_data:
            cleaned_text = self.parsed_data.get('content')
        else:
            raise ValueError("No data to clean.")

        for method in method_list:
            if method == 'abnormal':
                cleaned_text = data_cleaner.AbnormalCleaner(cleaned_text).to_clean().get("text")
            elif method == 'filter':
                cleaned_text = data_cleaner.TextFilter(cleaned_text).to_filter()
                cleaned_text = cleaned_text.get("text") if cleaned_text else ''
            elif method == 'private':
                cleaned_text = data_cleaner.PrivacyDesensitization(cleaned_text).to_private().get("text")

        if self.parsed_data:
            origin_dict = self.parsed_data
            origin_dict['content'] = cleaned_text
            self.parsed_data = None
            return origin_dict
        else:
            return cleaned_text

    def get_pre_label(self,
                      api_key: str,
                      base_url: str,
                      model_name: str,
                      chunk_size: int = 500,
                      chunk_overlap: int = 100,
                      question_number: int = 5,
                      max_workers: int = 5,
                      messages: List[Dict[str, str]] = None):
        return generatr_qa_pairs(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            question_number=question_number,
            max_workers=max_workers,
            message=messages,
            file_path=self.file_path
        )

    def _parse_file(self, file_path):
        """
        Create a parser instance using ParserFactory and parse the file.

        :param file_path: The path to the file to be parsed.
        :return: The parsed data.
        """
        try:
            parser = ParserFactory.create_parser(
                file_path=file_path,
                to_markdown=self.to_markdown,
                timeout=self.timeout
            )
            if parser:
                return parser.parse(file_path=file_path)
        except Exception as e:
            raise e


if __name__ == '__main__':
    # pass

    data = DataMax(file_path=[r"C:\Users\cykro\Desktop\测试集20241230\Excel表格\Hi-Dolphin Q&A历史 2024年11月04日.xlsx"])
    data = data.get_data()
    print(data)
