import os
import warnings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")

# PyMuPDFLoader 을 이용해 PDF 파일 로드
loader = PyMuPDFLoader("C:/Users/jyhan/Desktop/2024/prometheus/promichatbot/rag/rag/labor_low.pdf")
pages = loader.load()


# 문서를 문장으로 분리
## 청크 크기 500, 각 청크의 50자씩 겹치도록 청크를 나눈다
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(pages)

#print(docs)

from langchain.embeddings import HuggingFaceEmbeddings

# 문장을 임베딩으로 변환하고 벡터 저장소에 저장
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cpu'},
    #model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore = Chroma.from_documents(docs, embeddings)

# 벡터 저장소 경로 설정
## 현재 경로에 'vectorstore' 경로 생성
vectorstore_path = 'C:/Users/jyhan/Desktop/2024/prometheus/promichatbot/rag/rag/vectorstore'
os.makedirs(vectorstore_path, exist_ok=True)


# 벡터 저장소 생성 및 저장
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
# 벡터스토어 데이터를 디스크에 저장
vectorstore.persist()
print("Vectorstore created and persisted")