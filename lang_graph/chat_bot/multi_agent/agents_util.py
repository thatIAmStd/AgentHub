"""agents相关工具类封装"""
import operator
from typing import TypedDict, Sequence, Annotated, Literal

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models.base import BaseChatOpenAI


# agent消息体
class AgentState(TypedDict):
    # 消息列表
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 消息发送者
    sender: str


# 创建智能体函数
def create_agent(llm: BaseChatOpenAI, tools, system_message):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """你是一个很有效的ai助手，擅长调用工具;如果你无法解决问题,则直接返回你的结果,会有其他助手来帮忙你解决问题。
        你只需要做好你自己的事就行了,然后把结果丢出去。如果你认为整个团队的工作都做完了，则在回答最后面加上FINAL JOB。
        而你有权限使用的工具如下：{tool_names}.\n{system_message}
        """),
            MessagesPlaceholder(variable_name="messages")
        ],

    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    print(prompt)

    return prompt | llm.bind_tools(tools)


# 创建智能体节点
def create_node(state, agent, name):
    print(state)
    result = agent.invoke(state)

    if isinstance(result, ToolMessage):
        print("工具节点")
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    return {
        "messages": [result],
        "sender": name
    }


# 路由器函数，用于决定下一步是执行工具还是结束任务
def router(state: AgentState) -> Literal["call_tool", "__end__", "excel_tool","chart_tool","continue"]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "call_tool"
    if "FINAL JOB" in last_message.content:
        return "__end__"

    if last_message.tool_calls[0] and 'excel_tool' in last_message.tool_calls[0]['name']:
        return "excel_tool"

    # 如果既没有工具调用也没有完成任务，继续流程，返回 "continue"
    return "continue"
