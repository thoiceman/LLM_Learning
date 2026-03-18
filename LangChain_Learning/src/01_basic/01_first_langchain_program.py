import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage


# 加载环境变量
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY=='your_groq_api_key_here':
    raise ValueError("请在.env文件中配置有效的GROQ_API_KEY")


# 初始化模型
model = init_chat_model("groq:llama-3.3-70b-versatile",api_key=GROQ_API_KEY)


# 示例一：基础的LLM调用
def example01():
    response=model.invoke('你好，你是谁？')
    print(response)

# 示例二：使用消息列表
def example02():
    # 构建消息列表
    message=[
        SystemMessage(content="你现在是一个专业的开发工程师"),
        HumanMessage(content="什么是高级语言？")
    ]
    response01=model.invoke(message)
    print(response01)
    # 继续对话，将AI的回答放入消息列表
    message.append(response01)
    message.append(HumanMessage(content="能给我举一个例子吗？"))
    response02=model.invoke(message)
    print(response02)

# 示例三：使用字典格式的消息
def example03():
    message=[
        {"role":"system","content":"你现在是一个专业的音乐鉴赏人"},
        {"role":"user","content":"请给我推荐一首流行音乐"}
    ]
    response01=model.invoke(message)
    print(response01)
    message.append(response01)
    message.append({"role":"user","content":"给我推荐一首摇滚音乐"})
    response02=model.invoke(message)
    print(response02)
def main():
    try:
        # example01()
        example03()
    except ImportError as e:
        print(f"运行出错:{e}")

if __name__ == '__main__':
    main()


