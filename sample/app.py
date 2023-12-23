from openai import OpenAI
import streamlit as st
import ss_chatbot.chatbot as cb
from langchain.chains import LLMChain

import time
DOCS = ['DIVISION OF ASSETS AFTER DIVORCE.txt', 'INHERITANCE.txt'] # Documents' Names

DOC_FULL_PATH = 'C:/Users/vpafu/Desktop/myFolder/Data Science/Text Mining/splitsmart/docs' # Documents path


with st.spinner('Wait for it...'):
    #time.sleep(5)
    s1 = cb.SingletonClass()
    for elem in DOCS:
        list_docs = s1.ingest_document(DOC_FULL_PATH, elem)
        print(elem)
        s1.chroma_embedding_documents(list_docs)
    s1.create_multiquery_retriever()
    s1.set_prompt()
    my_chain = s1.invoke_chain()

st.success('Done!')

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    my_chain.invoke(prompt)
    #msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)