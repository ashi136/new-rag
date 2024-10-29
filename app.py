import os
import warnings
import sqlite3
from datetime import datetime
import uuid
from flask import Flask, request, jsonify, render_template, session
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
app.secret_key = 'your-super-secret-key-here-replace-this'

# Suppress warnings
warnings.filterwarnings('ignore')

# Database setup
def setup_database():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    
    # Drop existing tables if they exist
    c.execute('DROP TABLE IF EXISTS conversations')
    c.execute('DROP TABLE IF EXISTS sessions')
    c.execute('DROP TABLE IF EXISTS users')
    
    # Create users table
    c.execute('''CREATE TABLE users
                 (id TEXT PRIMARY KEY,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create sessions table
    c.execute('''CREATE TABLE sessions
                 (id TEXT PRIMARY KEY,
                  user_id TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    # Create conversations table
    c.execute('''CREATE TABLE conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  message TEXT,
                  response TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (session_id) REFERENCES sessions(id))''')
    
    conn.commit()
    conn.close()

# Call setup_database when the application starts
setup_database()

def get_or_create_user():
    user_id = str(uuid.uuid4())
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (id) VALUES (?)', (user_id,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()
    return user_id

def get_user_conversation_history(user_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT s.id, s.created_at, c.message, c.response, c.timestamp 
                 FROM sessions s
                 LEFT JOIN conversations c ON s.id = c.session_id
                 WHERE s.user_id = ?
                 ORDER BY s.created_at DESC, c.timestamp ASC''', (user_id,))
    history = c.fetchall()
    conn.close()
    return history

def save_conversation(session_id, message, response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO conversations (session_id, message, response)
                 VALUES (?, ?, ?)''',
              (session_id, message, response))
    conn.commit()
    conn.close()

def create_new_session(user_id):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO sessions (id, user_id) VALUES (?, ?)',
              (session_id, user_id))
    conn.commit()
    conn.close()
    return session_id

# Initialize AI components
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", convert_system_to_human=True)

# Create directory if it doesn't exist
local_directory = "time_management_articles"
os.makedirs(local_directory, exist_ok=True)

def load_documents(urls, local_directory):
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading URL {url}: {str(e)}")
    
    if os.path.exists(local_directory):
        try:
            local_loader = DirectoryLoader(local_directory, glob="*.txt")
            documents.extend(local_loader.load())
        except Exception as e:
            print(f"Error loading local documents: {str(e)}")
    return documents

# List of URLs
urls = [
    "https://www.workast.com/blog/time-management-and-productivity-in-the-modern-workplace/",
    "https://blog.hubspot.com/marketing/mentor-tips-positive-impact",
]

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
   - Start by asking open-ended questions to help the user articulate their problem.
   - Listen carefully and ask clarifying questions.
   - Continue asking reflective questions to guide the user.

2. **Testing Understanding**:
   - After discussing solutions, offer practice scenarios.
   - Provide constructive feedback.

3. **Wrap-Up**:
   - Summarize progress and offer next steps.

Throughout, maintain a supportive and therapeutic tone.
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Create chains
question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

retriever_prompt = """
Given the chat history and the latest user question,
formulate a standalone question which can be understood without the chat history.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
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

def time_management_mentor(question: str, session_id: str = "default") -> str:
    try:
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return response["answer"]
    except Exception as e:
        print(f"Error in time_management_mentor: {str(e)}")
        return "I apologize, but I encountered an error processing your request."

@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = get_or_create_user()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        session['user_id'] = get_or_create_user()
    
    if 'session_id' not in session:
        session['session_id'] = create_new_session(session['user_id'])
    
    data = request.json
    user_input = data.get('message')
    if not user_input:
        return jsonify({'error': 'Message is required'}), 400
    
    response = time_management_mentor(user_input, session['session_id'])
    save_conversation(session['session_id'], user_input, response)
    
    return jsonify({'response': response})

@app.route('/history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        session['user_id'] = get_or_create_user()
    
    history = get_user_conversation_history(session['user_id'])
    return jsonify({'history': history})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    if 'user_id' not in session:
        session['user_id'] = get_or_create_user()
    
    new_session_id = create_new_session(session['user_id'])
    session['session_id'] = new_session_id
    
    # Clear the conversation store for the new session
    if session.get('session_id') in store:
        del store[session['session_id']]
    
    return jsonify({
        'status': 'success',
        'message': 'Session reset successfully',
        'new_session_id': new_session_id
    })

if __name__ == '__main__':
    app.run(debug=True)