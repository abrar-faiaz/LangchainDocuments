# Import os to set API key
import os
# Import Hugging Face transformers and other necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFaceLLM
from langchain.embeddings import HuggingFaceEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set API key for Hugging Face Service
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'yourhuggingfaceapikeyhere'

# Load Hugging Face model and tokenizer
model_name = "gpt2"  # You can change this to any model you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create instance of Hugging Face LLM
llm = HuggingFaceLLM(model=model, tokenizer=tokenizer, temperature=0.1, verbose=True)
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('🦜🔗 GPT Investment Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content)
