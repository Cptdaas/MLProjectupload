# Streamlit Configuration
st.set_page_config(
    page_title="RAG",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme Configuration
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background: #333;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Retrieval Augmented Generation Engine")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever
