"""
For this agent, we will use Langchain and huggingface, Groq for QA.
"""
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key = groq, model="llama-3.3-70b-versatile")
os.environ["GROQ_API_KEY"] = groq
loader = PyMuPDFLoader("./resume.pdf")

docs = loader.load()

text_spiltter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap = 64)
spilts = text_spiltter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

vector_store = Chroma.from_documents(documents=spilts, embedding=embeddings)


retreiver = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the following question in a detailed manner based only on the provided context:

Context: {context}

Question: {question}

Answer the question using only the information from the provided context. If you cannot answer the question based on the context, please say so.""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retreiver | format_docs , "question": RunnablePassthrough()}
    | prompt
    |llm
    | StrOutputParser()
)

# response = rag_chain.invoke("summarize this document")
# print(response)