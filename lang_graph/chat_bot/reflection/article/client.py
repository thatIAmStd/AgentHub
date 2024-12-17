from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import add_messages, StateGraph
from langgraph.graph import END
from agents import writer, teacher
from lang_graph.chat_bot import display

MaxRound = 6


class State(TypedDict):
    messages: Annotated[list, add_messages]


# 添加节点
def writer_node(state: State) -> State:
    return {"messages": [writer.invoke(state["messages"])]}


def teacher_node(state: State) -> State:
    msg_cls = {"ai": HumanMessage, "human": AIMessage}
    # 处理消息，保持用户的原始请求（第一个消息），转换其余消息的类型
    # 这里用了一个列表推导式，作用是创建一堆 ai 和 human对换的list
    content = [state["messages"][0]] + [
        msg_cls[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    # return {"messages": [teacher.invoke(content)]}
    # 调用反思器(reflect)，将转换后的消息传入，获取反思结果
    res = teacher.invoke(content)

    # 返回新的状态，其中包含反思后的消息
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(State)
builder.add_node("writer", writer_node)
builder.add_node("teacher", teacher_node)


# 添加边
def router(state: State):
    if len(state["messages"]) > MaxRound:
        return END

    return "teacher"


builder.add_edge(START, "writer")
builder.add_conditional_edges("writer", router)
builder.add_edge("teacher", "writer")

# 编译图,增加记忆
saver = MemorySaver()
graph = builder.compile(checkpointer=saver)
# display.save_img("graph.png",graph.get_graph().draw_mermaid_png())

round = 0
inputs = {
    "messages": [
        HumanMessage(content="用鲁迅阿Q正传的风格，来写一下西游记三大白骨精")
    ],
}
config = {"configurable": {"thread_id": "1"}}

for event in graph.stream(inputs,config):
    round = round + 1
    print(f"---------------------------round:{round}------------------------------------------")
    print(event)
