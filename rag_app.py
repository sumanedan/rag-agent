import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# ---- STEP 1: Load your PDF ----
print("📄 Loading PDF...")
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# ---- STEP 2: Split into chunks ----
print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# ---- STEP 3: Embed and store in vector DB ----
print("🧠 Creating embeddings and storing in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)
print("✅ Vector database ready!")

# ---- STEP 4: Create retriever and chain ----
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

prompt = PromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the document."

Context: {context}

Question: {question}

Answer:
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- STEP 5: Ask questions ----
print("\n📚 RAG Chatbot ready! Ask questions about your PDF.")
print("Type 'quit' to exit")
print("-" * 40)

while True:
    question = input("You: ")
    if question.lower() == "quit":
        print("Bye! 👋")
        break

    response = chain.invoke(question)
    print(f"AI: {response}")
    print("-" * 40)