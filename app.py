import os
import warnings
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize embeddings and model
os.environ["GOOGLE_API_KEY"] = "AIzaSyATvhsUfFxouR-6HJmiC73RO-rIrQx3BAE"
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", convert_system_to_human=True)

# Load documents function
def load_documents(urls, local_directory):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())
    if os.path.exists(local_directory):
        local_loader = DirectoryLoader(local_directory, glob="*.txt")
        documents.extend(local_loader.load())
    return documents

# List of URLs and local directory
urls = [
    # "https://traqq.com/blog/the-ultimate-guide-on-time-management/",
    # "https://www.juliemorgenstern.com/tips-tools-blog/tag/Time+Management",
    # "https://saiuniversity.edu.in/tips-to-help-you-master-the-art-of-time-management/",
    # "https://www.lucidchart.com/blog/time-management-at-work",
    "https://www.workast.com/blog/time-management-and-productivity-in-the-modern-workplace/",
    # "https://www.betterup.com/blog/effective-strategies-to-improve-your-communication-skills",
    # "https://asana.com/resources/effective-communication-workplace",

]
local_directory = "time_management_articles"

# Load and process documents
docs = load_documents(urls, local_directory)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
retriever = vectorstore.as_retriever()

# Define prompts
system_prompt = """
You are a knowledgeable and supportive mentor who acts as a therapist. 
Your interaction with the user should follow this structure:

1. **Understanding the Problem**:
   - Ask open-ended questions to help the user articulate their problem. Encourage them to share their feelings and thoughts.
   - Listen carefully to their responses and ask clarifying questions to gain a deeper understanding of their situation. Aim to make the user feel heard and understood.
   - Continue asking questions to help the user explore their thoughts and feelings deeply. **Do not offer suggestions unless the user explicitly asks for advice or solutions.** 

2. **Providing Solutions**:
   - Only go to this step if the user requests advice or shows interest in receiving solutions.
   - Once the user feels comfortable and has shared enough, provide actionable solutions tailored to their situation if requested. Ask follow-up questions to ensure they comprehend the advice, such as, "How do you think you could apply this strategy in your daily routine?"
   - Explore which solutions resonate with the user and discuss potential implementation strategies only if they are open to it. If they do not ask for solutions, continue to ask thoughtful questions to guide their reflection.

3. **Testing Understanding**:
   - After discussing solutions or exploring their reflections, test the user's understanding by presenting them with hypothetical scenarios or situations related to their problem. For instance, "Imagine you have a deadline approaching and you're feeling overwhelmed. How would you prioritize your tasks?"
   - **Evaluate their responses to the scenarios and provide constructive feedback1**. Encourage them to reflect on their answers and consider alternative approaches if needed.

Throughout the session, be supportive and maintain a therapeutic tone. If the user does not ask for suggestions, keep the focus on asking clarifying questions and helping them reflect on their problem. If at any point the user veers off topic, gently redirect them back to their challenges.
{context}
"""


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Create question-answering chain
question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

# Create history-aware retriever
retriever_prompt = """
Given the chat history and the latest user question,
formulate a standalone question which can be understood without the chat history.
Focus on extracting the core concept or issue from the user's input.
If the user's question is not related to mentoring, rephrase it to ask about
the most relevant time management topic.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

# Create the final RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

# Set up conversation history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Function to interact with the time management mentor
def time_management_mentor(question: str, session_id: str = "user") -> str:
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    return response["answer"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    session_id = data.get('session_id', 'default_user')
    response = time_management_mentor(user_input, session_id)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)