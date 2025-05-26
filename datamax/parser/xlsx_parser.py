import multiprocessing
import time
from multiprocessing import Queue
from datamax.parser.base import MarkdownOutputVo
from datamax.parser.base import BaseLife
from openpyxl import load_workbook
import warnings

warnings.filterwarnings("ignore")

class XlsxParser(BaseLife):
    def __init__(self, file_path, timeout):
        super().__init__()
        self.file_path = file_path
        self.timeout = timeout

    def _parse(self, file_path: str, result_queue: Queue) -> dict:
        from markitdown import MarkItDown
        markitdown = MarkItDown()  # 子进程内部初始化

        try:
            wb = load_workbook(
                filename=file_path,
                data_only=True,
                read_only=True
            )
            wb.close()
        except Exception as e:
            raise e

        mk_content = markitdown.convert(file_path).text_content
        lifecycle = self.generate_lifecycle(
            source_file=file_path,
            domain="Technology",
            usage_purpose="Documentation",
            life_type="LLM_ORIGIN"
        )
        output_vo = MarkdownOutputVo(self.get_file_extension(file_path), mk_content)
        output_vo.add_lifecycle(lifecycle)
        result_queue.put(output_vo.to_dict())
        time.sleep(0.5)
        return output_vo.to_dict()

    def parse(self, file_path: str) -> dict:
        result_queue = Queue()
        process = multiprocessing.Process(
            target=self._parse,
            args=(file_path, result_queue)
        )
        process.start()
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            print(f"plz waiting...: {int(time.time() - start_time)}")
            if not process.is_alive():
                break
            if not result_queue.empty():
                return result_queue.get()
            time.sleep(1)
        else:
            process.terminate()
            process.join()
            raise TimeoutError(f"Parsing timed out after {self.timeout} seconds")