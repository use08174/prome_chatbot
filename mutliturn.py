from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Memory 객체 생성
memory = ConversationBufferMemory(return_messages=True)

# prompt + model + output parser
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are human. Answer the question"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

llm = ChatUpstage(api_key="up_XH5nucsIByalKYtgLxSRlCaWxqTIq", model="solar-pro")
output_parser = StrOutputParser()

# 체인 방식으로 LLM 연결
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# 멀티턴 대화 시나리오
def run_conversation(input_text):
    response = chain.predict(input=input_text)  # input 키워드 인자 추가
    print(f"질문: {input_text}")
    print(f"답변: {response}\n")
    print(f"현재 대화 기록: {memory.buffer}") # 현재 대화 기록 출력

# 대화 시나리오 실행
run_conversation("Hi. Long time no see")
run_conversation("What did you do yesterday?")
run_conversation("What did you do yesterday?")