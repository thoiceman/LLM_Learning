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


# 示例四：相关参数配置
"""
- init_chat_model 常用参数
    temperature:生成内容的随机性
    max_tokens:生成内容的最大token长度
    model_kwargs:传递给底层模型的额外参数
"""
def example04():
    model_deterministic=init_chat_model("groq:llama-3.3-70b-versatile",temperature=0.0,max_tokens=100)
    prompt='创作一首关于西湖的诗句'
    # 连续调用两次，观察输出内容的一致性
    for i in range(2):
        response=model_deterministic.invoke(prompt)
        print(response)
    
    model_deterministic=init_chat_model("groq:llama-3.3-70b-versatile",temperature=1.5,max_tokens=100)
    prompt='创作一首关于西湖的诗句'
    # 连续调用两次，观察输出内容的一致性
    for i in range(2):
        response=model_deterministic.invoke(prompt)
        print(response)

# 示例五：了解invoke方法的返回值结构
"""
invoke()返回的结构如下：
content:恢复的文本内容
response_metadata:响应元数据（如token使用量，模型信息）
additional_kwargs:（额外的关键字参数）
id:（消息id）
"""
def example05():
    prompt='介绍一下周杰伦'
    response=model.invoke(prompt)
    print(response.content)
    print(response.response_metadata)
    print(response.additional_kwargs)
    print(response.id)



# 示例六：错误处理
def example06():
    try:
        response=model.invoke('杭州有什么著名的旅游景点？')
        print(response)

    except ValueError as e:
        print(f"配置错误：{e}")
    except ConnectionError as e:
        print(f"连接错误{e}")
    except Exception as e:
        print(f"未知错误：{e}")









def main():
    try:
        # example01()
        example05()
    except ImportError as e:
        print(f"运行出错:{e}")

if __name__ == '__main__':
    main()


