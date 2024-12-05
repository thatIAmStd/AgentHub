from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from sqlalchemy import values
from typing_extensions import Annotated
from langgraph.graph import message, StateGraph


# 定义状态机
class State(TypedDict):
    msg: Annotated[List, message.add_messages]


# 定义Graph
graph_builder = StateGraph(State)

# 定义机器人
chat_model = ChatOpenAI(model="gpt-4o-mini")


def chat_bot(state: State):
    return {"msg": [chat_model.invoke(state["msg"])]}


# 添加节点，定义边,编译图形
graph_builder.add_node("chat_bot", chat_bot)
graph_builder.add_edge(START, "chat_bot")
graph_builder.add_edge("chat_bot", END)
graph = graph_builder.compile()

# 接受用户输入
while True:
    user_input = input("用户:")
    if user_input in ["quit", "q"]:
        print("chat over")
        break

    for event in graph.stream({"msg": ("user", user_input)}):

        for value in event.values():
            print("ai response:" + value["msg"][-1].content)
