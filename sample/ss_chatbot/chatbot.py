from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter


import os
import shutil



class SingletonClass:

    _self = None
    _embedding_model = None # Model used for the Embedding
    _vector_db = None # Local vector DB
    _K = 5
    _llm = None # LLM used for generating the response
    _retriever = None # Vector DB retriever
    _chain = None # Execution chain
    _remove_local_vector_db_flag = False # Indicates if it's necessary to drop the old Vector DB (if exists)

    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)            
        return cls._self

    def __init__(self, **kwargs):
        
        self._embedding_model = HuggingFaceEmbeddings(model_name=kwargs['EMBEDDING_MODEL'])
        self._vector_db_path = kwargs['VECTOR_DB_PATH']
        self._llm=OpenAI(temperature=0, openai_api_key=kwargs['OPEN_AI_KEY'])
        self._remove_local_vector_db_flag = bool(kwargs['REMOVE_OLD_VECTOR_FLAG'])
        self._prompt_path = kwargs['PROMPT_PATH']

        # Remove existing local DB vector
        if self._remove_local_vector_db_flag:
            shutil.rmtree(self._vector_db_path) if os.path.exists(self._vector_db_path) else print("{} path doesn't exist".format(self._vector_db_path))

        # Create a new folder for a local DB Vector if not exists
        os.mkdir(self._vector_db_path) if not os.path.exists(self._vector_db_path) else print("{} path already exist".format(self._vector_db_path))
        
        print('Initializated')

    def _create_document(cls, 
                         article: str, 
                         doc_name: str
                         ) -> Document:
        '''
        Description:
        ----------------------
        Creates a Document with the following structure:

        {\n
            page_content:\n
                        "text"\n
            metadata:{\n
                        "doc_name" -> name of the Document from which the article comes \n
                        "article_number" -> Number of the article of the civil code \n
                        "title" -> Title of the article contained in page_content \n
            }\n
        }\n

        Input:
        ---------------------
        article: (str) -> Article of the Civil Code
        doc_name: (str) -> Name of the Document from which the article comes

        Output:
        ---------------------
        : (Document) -> with the structure described above

        '''

        _ = {
            "doc_name": doc_name.strip(),
            "article_number": article.split('\n')[1].strip(),
            "title": article.split('\n')[2].strip()
        }
        
        return Document(page_content=article, metadata=_)

    def ingest_document(cls, 
                        doc_folder_path: str, 
                        document_name: str
                        ) -> list[Document]:
        '''
        Description:
        ----------------------
        Read the Document (.txt) specified in document_name 
        variable, then it will be splitted into several articles 
        of the civil code.
        Finally, each article will be transformed into a single 
        Document enriched with some metadata information.

        Input:
        ---------------------
        doc_folder_path: (str) -> Article of the Civil Code
        document_name: (str) -> Name of the Document from which the article comes

        Output:
        ---------------------
        documents_formatted: (list[Document]) -> Where each Document is an article from Civil Code
        '''
        documents_formatted = []
        loader = TextLoader(file_path=doc_folder_path + '/' + document_name)
        document_ingested = loader.load() # Load the document
        articles_list = document_ingested[0].page_content.split('\n\n') # Split the document into several Civil Code articles
        
        # create a Document object for each article in the document loaded
        
        documents_formatted = map(lambda x: cls._create_document(article=x,
                                                                 doc_name=document_name),
                                  articles_list
                                  )
        del loader, document_ingested, articles_list

        return list(documents_formatted)
    
    def load_chroma_persistent_db(cls):
        """
        Description:
        ----------------------
        Retrieve the Chroma local vector DB.
        """
        cls._vector_db = Chroma(persist_directory=cls._vector_db_path, 
                                embedding_function=cls._embedding_model)

    def chroma_embedding_documents(cls, 
                            doc_list: list[Document]
                            ) -> None:
            """
            Description:
            ---------------------
            This method performs the embedding over the Documents given in input by
            using the Class Embedding Model and stores the embeddings into a local
            Chroma vector DB

            Input:
            ---------------------
            doc_list: (list[Document]) -> List of the Documents to embed
            """

            # Embed documents and store them into a local Chroma vector DB 

            cls._vector_db = Chroma.from_documents(documents=doc_list,       # List of the documents
                                          embedding=cls._embedding_model,    # embedding model
                                          persist_directory=cls._vector_db_path          # path to existing db local directory 
                                          )

    def vector_search(cls, query):

        doc_results = cls._vector_db.similarity_search_with_score(query=query, 
                                                                  k=cls._K)
        return list(doc_results)

    def create_multiquery_retriever(cls):
        """
        Description:
        ---------------------
        This method creates a multiquery vector db retriever 
        by setting it up with the class paramters that follow:
        - vector_db
        - llm
        """
        if cls._retriever is None:
            cls._retriever = MultiQueryRetriever.from_llm(
                retriever=cls._vector_db.as_retriever(), 
                llm=cls._llm,
                include_original=True
                )
            print('Retriever created')
        else:
            print('Retriever already exists.')

    def set_prompt(cls):
        """
        Description:
        ---------------------
        This method set up the prompt for the LLM
        """
        with open(cls._prompt_path, "r") as file:
            prompt = file.read()

        cls._prompt = PromptTemplate.from_template(str(prompt))

    def translate_input(self, 
                         input: str
                         ) -> str:
        """
        Description:
        ---------------------
        This methos translate the input into English,
        the language of the embedded documents

        Input:
        ---------------------
        input: (str) -> String to translate in English

        Output:
        ---------------------
        : (str) -> String Translated
        """

        prompt = """
        translate in English the following input:
        {}
        """.format(input)

        response = self._llm.invoke(prompt)

        return response
        
    def create_chain(cls) -> LLMChain:
        """
        Description:
        ---------------------
        This method creates a chain to: \n
        1 - Take the input query \n
        2 - Retrieve the most similar documents embedded \n
        3 - Integrate them into the prompt\n
        4 - LLM call for generating the response


        Output:
        ---------------------
        : (LLMChain) -> LLM Chain to invoke
        """

        output_parser = StrOutputParser()

        setup_and_retrieval = RunnableParallel(
            {
             "context": cls._retriever, 
             "question": RunnablePassthrough()
             }
        )

        chain = setup_and_retrieval | cls._prompt | cls._llm | output_parser

        return chain

    