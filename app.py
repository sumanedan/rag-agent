import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import tempfile

# ---- Load API keys ----
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# ---- Page config ----
st.set_page_config(
    page_title="RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# ---- Sidebar ----
with st.sidebar:
    st.title("🤖 RAG Agent")
    st.markdown("---")

    # PDF Upload
    st.subheader("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    st.markdown("---")
    st.subheader("🛠️ Tools Active")
    st.success("🔍 Web Search")
    st.success("🧮 Calculator")
    if uploaded_file:
        st.success("📄 PDF Search")
    else:
        st.warning("📄 PDF Search (upload PDF)")

    st.markdown("---")
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Main area ----
st.title("💬 AI Research Agent")
st.caption("Ask me anything — I can search the web, read your PDF, and do math!")

# ---- Initialize chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Display chat history ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Build tools ----
@st.cache_resource
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def get_search_tool():
    return TavilySearchResults(max_results=3)

@tool
def calculator(expression: str) -> str:
    """Useful for math calculations. Input should be a math expression like '299 * 12'."""
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_resource
@st.cache_resource
def build_pdf_tool(file_path: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        chunks, embeddings
        # No persist_directory — runs in memory
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Search the uploaded PDF document for relevant information."
    )

# ---- Build agent ----
def get_agent(pdf_file=None):
    tools = [get_search_tool(), calculator]
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            tmp_path = f.name
        pdf_tool = build_pdf_tool(tmp_path)
        tools.append(pdf_tool)

    # System prompt guides the agent on when to use which tool
    system_prompt = """You are a helpful AI assistant with access to tools.
    
Rules:
- For math → use calculator
- For PDF/document questions → use pdf_search  
- For current/live information → use web search
- Always give a final answer within 3 tool calls
- Do not repeat the same tool call twice
- If you have enough information, stop and answer immediately"""

    return create_react_agent(get_llm(), tools, prompt=system_prompt)

# ---- Chat input ----
if prompt := st.chat_input("Ask anything about your PDF or the web..."):

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
       with st.spinner("🤔 Agent thinking... (may take 10-15 seconds)"):
            try:
                agent = get_agent(uploaded_file)
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"recursion_limit": 25}  # prevents infinite loops
                )

                # Extract answer
                last_message = response["messages"][-1].content
                if isinstance(last_message, list):
                    answer = " ".join(
                        block["text"] for block in last_message
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    answer = last_message

                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)