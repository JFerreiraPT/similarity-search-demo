from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()



#define our model for embeddings
hf_embeddings = HuggingFaceEmbeddings()


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





#3 Receive chunks, generate embeddings and then  #4 store indexes
def create_embeddings(documents, index_name):

    #create embeddings and index with FAISS help
    #Just pass embeddings algorithm we want to use
    vectorStore = FAISS.from_documents(documents, hf_embeddings)

    #We can save it locally for future use.
    vectorStore.save_local(index_name)
    return vectorStore




#5 ask pdf
def ask_store(vectorStore, query):
    print("Asking store")

    #Langchain abstraction for QA
    #We need to pass the LLM instance that we want to use
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    #Similarity search in the vector database
    #The top 5 rank documents are then returned
    relevant_docs = vectorStore.similarity_search(query, 5)

    # We will then run ou chain, the resulting documents are sent to the LLM as context, 
    # the LLM then interpret the question and answers it based on the provided context
    refined_answer = chain.run(input_documents=relevant_docs, question=query) 

    return refined_answer




def chat(vector_store):
    while True:
        question = input("Question: ")
        
        answer = ask_store(vector_store, question)
        print(answer)



def program():
    # If the vector database is already created we will use it
    # Otherwise, we will create it from documents

    try:
        vector_store = FAISS.load_local("ask_pdf_index", hf_embeddings)
    except:
        vector_store = create_embeddings(load_documents_pdf(), "ask_pdf_index")
        
    chat(vector_store)



program()