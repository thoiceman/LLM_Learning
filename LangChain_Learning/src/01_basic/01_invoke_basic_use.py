import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

# 加载环境变量
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY=='your_groq_api_key_here':
    raise ValueError('请在.env文件中配置有效的GROQ_API_KEY')

# 初始化模型
model=init_chat_model("groq:llama-3.3-70b-versatile",api_key=GROQ_API_KEY)



"""
理解invoke的返回值AIMessage对象
- 主要内容:content
- 消息ID:id
- 响应元数据:response_metadata
response_metadata里面还有模型相关的信息(model_name,model_provider),Token使用情况(prompt_tokens,completion_tokens,total_tokens)等

"""

# 示例一：invoke返回值
def example01():
    response=model.invoke("你好，你是谁？")
    print(response.content)
    print(response.id)
    print(response.response_metadata)

"""
理解invoke的三种输入格式
- 纯字符串
- 字典列表
- 消息对象
"""

# 格式一：纯字符串
def example02():
    response=model.invoke("你好，你是谁？")
    print(response.content)

# 格式二：字典列表
def example03():
    message=[
        {"role":"system","content":"你是一个个人助手，请尽量简洁回答问题"},
        {"role":"user","content":"一个成年人一天要摄入多少蛋白质？"}
    ]
    response=model.invoke(message)
    print(response.message)


# 格式三：消息对象
from langchain_core.messages import SystemMessage,HumanMessage
def example04():
    message=[
        SystemMessage(content="你是一个诗人"),
        HumanMessage(content="请给我写一首关于西湖的诗")
    ]
    response=model.invoke(message)
    print(response.content)


"""
理解系统角色的作用
- 通过不同的系统提示，让AI扮演不同角色
"""

def example05():

    question="1+1=?"

    # 幼儿园老师
    message=[
        {"role":"system","content":"你是一个幼儿园老师，请用通俗易懂的语言回答问题，并举例"},
        {"role":"user","content":question}
    ]
    response01=model.invoke(message)
    print(response01.content)

    # 数学专业教授
    message=[
        {"role":"system","content":"你是一个数学专业教授，请用通俗易懂的语言回答问题，并举例"},
        {"role":"user","content":question}
    ]
    response02=model.invoke(message)
    print(response02.content)

"""
实现多轮对话，理解对话历史
"""

def example06():
    conversation=[
        {"role":"system","content":"你是一个和蔼的朋友"}
    ]

    #第一轮对话
    conversation.append({"role":"user","content":"我今天很高兴，我们今天可以一起出去玩吗？"})
    response01=model.invoke(conversation)
    conversation.append({"role":"friend","content":response01.content})