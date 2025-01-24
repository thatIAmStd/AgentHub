from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai.chat_models import ChatOpenAI

model = "gpt-4o-mini"

java_planer = (
        ChatPromptTemplate.from_template(
            "用java代码实现{context}功能。不给出解释，只要代码部分。不要markdown格式，但是要保留换行符")
        | ChatOpenAI(model=model)
        | StrOutputParser()
)

python_planer = (
        ChatPromptTemplate.from_template(
            "用python代码实现{context}功能。不给出解释，只要代码部分。不要markdown格式，但是要保留换行符")
        | ChatOpenAI(model=model)
        | StrOutputParser()
)

chain = RunnableParallel(
    java_code=java_planer,
    python_code=python_planer
)
result = chain.invoke({"context": "快速排序"})

print(type(result))
print("java代码实现:")
print(result["java_code"])

print("python代码实现:")
print(result["python_code"])
