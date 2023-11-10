
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import openai
from langchain.llms import OpenAI

def read_pdf(pdf):

    # # For Single file
    # pdf_reader = PdfReader(pdf)
    # content = ''
    # for page in pdf_reader.pages:
    #     content += page.extract_text()

    # For multiple files
    content = ''
    if len(pdf):
        for i in range(len(pdf)):
           pdf_reader = PdfReader(pdf[i])
           for page in range(len(pdf_reader.pages)):
              content += pdf_reader.pages[page].extract_text()
    else:
        pdf_reader = PdfReader(pdf)
        content = ''
        for page in pdf_reader.pages:
            content += page.extract_text()
    
    return content

def text_splitter(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = splitter.split_text(text)
    return chunks

def vectore_store(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    knowledge_base = FAISS.from_texts(chunks, embedding=embeddings)
    return knowledge_base

def semantic_serch(vectoreStore,query):
    relevant_docs = vectoreStore.similarity_search(query,k=2) 
    return relevant_docs


def get_answer(query,relevant_docs):
      
    llm=HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":0.01, "max_length":1024})
    # llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=relevant_docs, question=query)
    return response
