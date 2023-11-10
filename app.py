from dotenv import load_dotenv
import streamlit as st
import utils as ut

load_dotenv()

st.set_page_config(page_title='Ask your PDF')
st.header('Ask your PDF📚')

# Upload the file / files
pdf = st.file_uploader('Upload your file here:', accept_multiple_files=True)

if pdf:
    
    # Extract the text
    text = ut.read_pdf(pdf)

    # Split into chunks
    chunks = ut.text_splitter(text)

    # create embeddings
    vectoreStore = ut.vectore_store(chunks)

    # User Query
    query = st.text_input("Ask a Question from the pdf:")

    if query:

        relevant_docs = ut.semantic_serch(vectoreStore=vectoreStore,query=query)

        response = ut.get_answer(query=query,relevant_docs=relevant_docs)
        
        st.write(response)
        st.write("Relevent DocumentsDocuments:books:")
        st.write(relevant_docs)
