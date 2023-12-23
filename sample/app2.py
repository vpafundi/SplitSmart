from openai import OpenAI
import streamlit as st
from ss_chatbot import chatbot as cb
DOCS = ['DIVISION OF ASSETS AFTER DIVORCE.txt', 'INHERITANCE.txt'] # Documents' Names

DOC_FULL_PATH = 'C:/Users/vpafu/Desktop/myFolder/Data Science/Text Mining/splitsmart/docs' # Documents path
DB_PATH = './Chroma_db/'
PROMPT_PATH = './prompt/lawyer.txt'
embedding_model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
open_ai_key = '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'

starting_msg = """
Ciao, sono il tuo assistente personale! se hai bisogno di informazioni 
relative alla gestione di eredità e divisione dei beni in contesti di 
divorzio matrimoniale cercherò di darti una mano!
"""


client = OpenAI(api_key='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')

st.title("Divorce Assistant")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

if 'first_execution' not in st.session_state:
    with st.spinner('Wait for it...'):
        s1 = cb.SingletonClass(EMBEDDING_MODEL=embedding_model,
                               OPEN_AI_KEY=open_ai_key,
                               VECTOR_DB_PATH=DB_PATH, 
                               REMOVE_OLD_VECTOR_FLAG=False,
                               PROMPT_PATH=PROMPT_PATH
                               )
        print('Singleton Created')
        for elem in DOCS:
            list_docs = s1.ingest_document(DOC_FULL_PATH, elem)
            #print(elem)
            s1.chroma_embedding_documents(list_docs)
            print(f'Documento {elem} embedded')
    del s1
    #s1.create_multiquery_retriever()
    #s1.set_prompt()
    #print('prompt set')
    #print('chain created')
    st.session_state['first_execution'] = True
    st.success('Done!')
else:
    print('already set up')


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": f"{starting_msg}"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])

if prompt := st.chat_input("Che domanda vuoi pormi?"):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    s1 = cb.SingletonClass(EMBEDDING_MODEL=embedding_model,
                               OPEN_AI_KEY=open_ai_key,
                               VECTOR_DB_PATH=DB_PATH, 
                               REMOVE_OLD_VECTOR_FLAG=False,
                               PROMPT_PATH=PROMPT_PATH
                               )
    s1.load_chroma_persistent_db()
    s1.create_multiquery_retriever()
    s1.set_prompt()
    my_chain = s1.invoke_chain()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        #for response in client.chat.completions.create(
        #    model=st.session_state["openai_model"],
        #    messages=[
        #        {"role": m["role"], "content": m["content"]}
        #        for m in st.session_state.messages
        #    ],
        #    stream=True,
        #):
        #for response in my_chain.stream(st.session_state.messages['content']):
        #    print(response, end="", flush=True)
        print(st.session_state.messages[-1]['content'])
        msg = st.session_state.messages[-1]['content']
        response = my_chain.invoke(msg)
        response = response.replace('Risposta: ', '')
        print(response)
        #full_response += (response.choices[0].delta.content or "")
        #message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})