from typing import Annotated, Sequence

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

import pandas as pd

# Python REPL 工具，用于执行 Python 代码
repl = PythonREPL()


@tool
def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL JOB"
    )


@tool
def excel_tool(
        data: Annotated[list, "The data to generate your excel"]
):
    """
    导出数据为 Excel 文件，只有一个入参data
    data为列表对象，第一列是excel表格的标题
    """
    export_to_excel(data)


def export_to_excel(
        data,
        file_path="agent_excel.xlsx",
        sheet_name="Sheet1"
):
    """
    导出数据为 Excel 文件

    :param data: 数据（列表或 DataFrame 格式）
    :param file_path: 导出的文件路径（如 output.xlsx）
    :param sheet_name: Excel 工作表名称，默认 Sheet1
    """
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("数据格式不支持，仅支持列表或 DataFrame 格式！")

        df.to_excel(file_path, index=False, sheet_name=sheet_name)
        print(f"数据成功导出到 Excel 文件: {file_path}")
    except Exception as e:
        print(f"导出 Excel 文件失败: {e}")


def save_img(location, img_bytes):
    try:
        output_file = location  # 可以根据需要修改文件名或路径
        with open(output_file, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        print(f"Error generating graph: {e}")


"""llm对应工具"""
# 搜索工具
search_tool = TavilySearchResults(max_results=5)
# python沙箱
python_tool = python_repl
excel_tool = excel_tool

if __name__ == '__main__':
    data_list = [
        ["Name", "Age", "City"],
        ["Alice", 25, "New York"],
        ["Bob", 30, "Los Angeles"],
        ["Charlie", 35, "Chicago"]
    ]
    export_to_excel(data_list)
