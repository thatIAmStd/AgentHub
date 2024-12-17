import functools

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from agents_util import *
from tools import search_tool, python_tool, excel_tool, save_img

search_agent = create_agent(
    ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    [search_tool],
    system_message="尝试回答问题,如果无法回答,不要凭空猜测，考虑通过搜索工具一次性查出结果。"
)

chart_agent = create_agent(
    ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    [python_tool],
    system_message="把接收到的数据,使用python函数,生成一个图表"
)

excel_agent = create_agent(
ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    [excel_tool],
        system_message="把接收到的数据,使用python函数,生成一个图表"
)

search_node = functools.partial(create_node, agent=search_agent, name="search_node")
chart_node = functools.partial(create_node, agent=chart_agent, name="chart_node")
excel_node = functools.partial(create_node, agent=excel_agent, name="excel_node")

tool_node = ToolNode([search_tool, python_tool,excel_tool], name="tool_node")

state = AgentState
workflow = StateGraph(state)
workflow.add_node("search_node", search_node, )
workflow.add_node("chart_node", chart_node)
workflow.add_node("excel_node", excel_node)
workflow.add_node("tool_node", tool_node)

# 定义条件边,搜索节点
workflow.add_conditional_edges(
    "search_node",
    router,
    {
        "call_tool": "tool_node",
        "continue": "chart_node",
        "__end__": END
    }
)
# 给chart节点添加条件边
workflow.add_conditional_edges(
    "chart_node",
    router,
    {
        "call_tool": "tool_node",
        "continue": "search_node",
        "__end__": END
    }
)
# 给chart节点添加条件边
workflow.add_conditional_edges(
    "excel_node",
    router,
    {
        "call_tool": "tool_node",
        "continue": "search_node",
        "__end__": END
    }
)
# 给工具节点添加条件边
workflow.add_conditional_edges(
    "tool_node",
    lambda x: x["sender"],
    {
        "search_node": "search_node",
        "chart_node": "chart_node",
        "excel_node": "excel_node"
    }
)
# 开始节点
workflow.add_edge(START, "search_node")
graph = workflow.compile()
save_img("graph_new.png", graph.get_graph(xray=True).draw_mermaid_png())

# 执行工作流

def do_action():
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="帮我查询一下美国2000年到2020年的每年GDP数据，并使用工具导出为excel格式"
                )
            ],
        },
        {"recursion_limit": 10},
        stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()  # 打印消息内容

do_action()


