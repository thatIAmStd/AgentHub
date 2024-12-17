from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

writer = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个写作助手，善于写出高质量的文章; 你的任务是根据用户的要求，写出富有哲理，内涵，和内容深度的文章。"
            "同时，你也会接受别人的反馈，并根据反馈做出对应的修改。以此来完成高质量文章的创作"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
) | ChatOpenAI(model="gpt-4o-mini", temperature=1.1)

teacher = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个老师，善于对别人文章进行评价，并给出有建设性的反馈意见。你的任务是，从文章的长度，内容深度，思想等方面进行评价。"
            "给出一个总的分数，所有评价项总分为10分。同时针对评价，你要给出反馈意见。好帮助他人提高作品质量"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
) | ChatOpenAI(model="gpt-4o-mini")


