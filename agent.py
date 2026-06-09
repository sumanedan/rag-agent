import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent

# ---- Load API keys ----
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# ---- Setup LLM ----
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ---- Tool 1: Web Search ----
search_tool = TavilySearchResults(max_results=3)

# ---- Tool 2: PDF Search ----
print("📄 Loading PDF knowledge base...")
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db_agent"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
pdf_tool = create_retriever_tool(
    retriever,
    name="techcorp_pdf_search",
    description="Search TechCorp employee handbook for company policies, products, salaries, work hours and contact info."
)

# ---- Tool 3: Calculator ----
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Useful for doing math calculations. Input should be a math expression like '25 * 4 + 10'."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# ---- Create Agent with all 3 tools ----
tools = [search_tool, pdf_tool, calculator]
agent = create_react_agent(llm, tools)

print("🤖 AI Agent ready! It can search the web, read your PDF and do math.")
print("Type 'quit' to exit")
print("-" * 40)

# ---- Chat loop ----
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bye! 👋")
        break

    print("🤔 Agent thinking...")
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    # Extract final answer
    last_message = response["messages"][-1].content
    if isinstance(last_message, list):
        # Extract just the text from the response
        final_answer = " ".join(
            block["text"] for block in last_message 
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        final_answer = last_message
    print(f"AI: {final_answer}")
    print("-" * 40)