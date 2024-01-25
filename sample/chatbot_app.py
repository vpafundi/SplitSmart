
from ss_chatbot import chatbot as cb
import time
import streamlit as st
import os



DB_PATH = './Chroma_db'
PROMPT_PATH = './utils/prompt/lawyer.txt'
STARTING_MSG_PATH = "./utils/starting_message.txt"
DOCS_PATH = './utils/docs/'
embedding_model = ''
llm = ''
openai_api_key = ''
#open_ai_key = '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'

DOCS = []
starting_msg = []


st.title("💬 SplitSmart")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

# Loading starting message
with open(STARTING_MSG_PATH, "r") as file:  
            starting_msg = file.read()

# If the path for the local vector DB doesn't exists, create it
if not os.path.exists(DB_PATH):
    print('no dir')
    os.mkdir(DB_PATH)

# If we didn't set up the variables yet
if 'setting_variable' not in st.session_state: 
    with st.container():
        # Select the model for embeddings
        embedding_model = st.selectbox(
             label='Choose a Model for embedding',
             options=['sentence-transformers/multi-qa-MiniLM-L6-cos-v1'],
             index=None,
             placeholder="Select a model...",
             key="Embedding model choose")
        # Select OpenAI LLM for creating the response
        llm = st.selectbox(
             label='Choose an OpenAI LLM',
             options=['gpt-3.5-turbo','gpt-4'],
             index=None,
             placeholder="Select a model...",
             key="LLM choose")
        # Insert OpenAI api key
        openai_api_key = st.text_input("OpenAI API Key", 
                                       key="chatbot_api_key", 
                                       type="password")

    while (not embedding_model or 
           not openai_api_key or 
           not llm):
        # while the variables are not set
        time.sleep(1)
    

    # Change state and set the variables as session_state variables
    st.session_state['setting_variable'] = True
    st.session_state['embedding_model'] = embedding_model
    st.session_state['llm'] = llm
    st.session_state['openai_api_key'] = openai_api_key
    

if 'first_execution' not in st.session_state and \
    len(os.listdir(DB_PATH +'/')) == 0:
       
    with st.spinner('Wait for it...'):

        info = st.info("Starting first execution")
        print("Starting first execution")

        # --------- Docs Loading ----------------
        doc_loading_info = st.info("Starting Documents loading")
        print("Starting message loaded")

        # Load the docs        
        for file in os.listdir(DOCS_PATH):
            if os.path.isfile(os.path.join(DOCS_PATH, file)):
                DOCS.append(str(file))

        doc_loading_info.empty()
        doc_success = st.success('Documents loaded', 
                                 icon="✅")
        # ----------- Singleton creation --------------

        print("Starting Singleton Creation")
        sing_creation = st.info("Starting SingletonCreation")

        s1 = cb.SingletonClass(EMBEDDING_MODEL=st.session_state['embedding_model'],
                               OPEN_AI_KEY=st.session_state['openai_api_key'],
                               OPEN_AI_MODEL=st.session_state['llm'],
                               VECTOR_DB_PATH=DB_PATH +'/', 
                               REMOVE_OLD_VECTOR_FLAG=False,
                               PROMPT_PATH=PROMPT_PATH
                               )
        
        sing_creation.empty()
        ding_success = st.success('Singleton created', 
                                  icon="✅")
        
        print('Singleton Created')

        # ----------- Document Embedding using the Singleton --------
        doc_embedding_info = st.info(f"Embedding Documents from {DOCS_PATH}")

        for elem in DOCS:
            list_docs = s1.ingest_document(DOCS_PATH, elem)
            s1.chroma_embedding_documents(list_docs)
            print(f'Documento {elem} embedded')

        doc_embedding_info.empty()
        doc_embedding_success = st.success('Documents Embedded', 
                                           icon="✅")

    
    # Delete vars
    del s1
    ding_success.empty()
    doc_success.empty()
    doc_embedding_success.empty()
    info.empty()

    # Set up first_execution Done in session_state variables
    st.session_state['first_execution'] = True

    done = st.success('Done!')
    time.sleep(3)
    done.empty()

else:
    print('Documents already embedded')


with st.sidebar:
    
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    st.info("LLM: {}".format(st.session_state['llm']))
    st.info("Embedding Model: {}".format(st.session_state['embedding_model']))
    st.info("Status OpenAI api key: {}".format("Loaded ✅" if st.session_state['openai_api_key'] is not None else "To Load"))


if "messages" not in st.session_state:

    st.session_state["messages"] = [{"role": "assistant", "content": f"{starting_msg}"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Che domanda vuoi pormi?"):

    if 'openai_api_key' not in st.session_state:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create or retrieve Singleton
    s1 = cb.SingletonClass(EMBEDDING_MODEL=st.session_state['embedding_model'],
                               OPEN_AI_KEY=st.session_state['openai_api_key'],
                               OPEN_AI_MODEL=st.session_state['llm'],
                               VECTOR_DB_PATH=DB_PATH + '/', 
                               REMOVE_OLD_VECTOR_FLAG=False,
                               PROMPT_PATH=PROMPT_PATH
                               )
    
    # Load existing local ChromaDB
    s1.load_chroma_persistent_db()
    # Create multiquery retriever
    s1.create_multiquery_retriever()
    # Set up prompt
    s1.set_prompt()
    # Create a chain with LLM chosen
    my_chain = s1.create_chain()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        print(st.session_state.messages[-1]['content'])
        msg = st.session_state.messages[-1]['content']
        print('msg: '+ msg)
        # Invoke the chain
        response = my_chain.invoke(msg)
        print(response)
        if 'Risposta: ' in response:
             response = response.split('Risposta: ')[1]
        #print(response[1])
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})