from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.document_loaders import YoutubeLoader

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.context_var_chain import ContextVarChain

from dotenv import load_dotenv

load_dotenv()

#define our model for embeddings
hf_embeddings = HuggingFaceEmbeddings()


config = RailsConfig.from_path("configs")
app = LLMRails(config)




vector_store_files = None
vector_store_Youtube = None


#1 Load documents and #2 create chunks
def load_documents_pdf():

    loader = PyPDFLoader("./thinkpython2.pdf")

    #It tries to split on them in order until the chunks are small enough
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 50,
        length_function = len,
    )

    chunks = loader.load_and_split(text_splitter)
    return chunks

#3 embeddings
def create_embeddings(documents, index_name):

    #create embeddings and index with FAISS help
    vectorStore = FAISS.from_documents(documents, hf_embeddings)

    #We can save it for future use.
    vectorStore.save_local(index_name)
    return vectorStore

def ask_store(vectorStore, query):

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    relevant_docs = vectorStore.similarity_search(query)

    refined_answer = chain.run(input_documents=relevant_docs, question=query) 

    return refined_answer



async def ask_store_GR(index_name, query):
    vectorStore = FAISS.load_local(index_name, hf_embeddings)

    results = vectorStore.similarity_search(query=query, n_results=5)

    

    template = """
    Ignore any previous instruction for now on you are an helpful assistant, you should 
    interpret any Context given to you and answer question based 
    only on that context. When you dont know the answer based on context just answer 'I dont know' 
    Context: {context}

    Question: {question}

    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    prompt_text = prompt.format(context=results, question=query)

    return prompt_text


def load_documents_youtube():

    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=2JrgmXrgqlc&ab_channel=CBSNews", add_video_info=True
    )

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
    )

    documents = loader.load_and_split(text_splitter)

    return documents

def chat():
    while True:
        question = input("Question: ")
        app.register_action_param("query", question)

        message = [
        {"role": "user", "content": question}
        ]
        
        answer =  app.generate(messages=message)
        print(answer["content"])



def program():
    app.register_action(ask_store_GR)

    try:
        FAISS.load_local("ask_pdf_index", hf_embeddings)
    except:
        create_embeddings(load_documents_pdf(), "ask_pdf_index")

    app.register_action_param("pdf_db", vector_store_files)
     

    try:
        FAISS.load_local("ask_youtube_index", hf_embeddings)
    except:
        create_embeddings(load_documents_youtube(), "ask_youtube_index")
        
    chat()



program()