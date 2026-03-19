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

