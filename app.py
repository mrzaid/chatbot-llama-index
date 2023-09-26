import os
import streamlit as st
from llama_index import  ServiceContext
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
from llama_index import SimpleDirectoryReader
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")


with st.sidebar:
    st.title("'🤗💬 Chat with your Data'")
    st.markdown('''
  
                ''')
def main():
    st.header("Chat with your Data")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the machine learning and your job is to answer technical questions."))
    index  = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query=st.text_input("Ask questions related to your Data")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)
if __name__=='__main__':
    main()    
ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the ML"))