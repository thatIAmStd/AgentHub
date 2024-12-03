import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# 加载民法典文档
b4_config = bs4.SoupStrainer(class_=("wp_articlecontent"))
doc_loader = WebBaseLoader(
    web_paths=("https://met.ntu.edu.cn/2024/0621/c9089a240361/pagem.htm",),
    bs_kwargs={"parse_only": b4_config}
)
doc = doc_loader.load()
print(len(doc[0].page_content))

# 文档分割
text_spliter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                              chunk_overlap=200,
                                              add_start_index=True)

docs = text_spliter.split_documents(doc)
print(len(docs))
# print(docs[0].page_content)

# 向量化存储
# vector_store = Chroma(
#     collection_name="min_fa_dian",
#     embedding_function=OpenAIEmbeddings(),
#     persist_directory="./min_fa_dian_db"
# )
# vector_store.add_documents(docs)
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)
print("embeeding完毕")

# 构建检索器，找出6个相似文档
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
re_docs = retriever.invoke("劳动仲裁")
for r in re_docs:
    print(r.page_content)
print("测试检索--------------------------")
# 生成回答
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer with chinese and give me the source from the context:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

for chunk in rag_chain.stream("拖欠工资怎么处理？"):
    print(chunk, end="", flush=True)
