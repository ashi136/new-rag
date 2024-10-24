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
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

# Suppress warnings
warnings.filterwarnings('ignore')

# Database setup
def setup_database():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  message TEXT,
                  response TEXT,
                  timestamp DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT UNIQUE,
                  created_at DATETIME)''')
    conn.commit()
    conn.close()

# Function to save a conversation to the database
def save_conversation(session_id, message, response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO conversations (session_id, message, response, timestamp)
                 VALUES (?, ?, ?, ?)''', (session_id, message, response, datetime.now()))
    conn.commit()
    conn.close()

# Function to get all conversation history
def get_all_conversation_history():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT s.session_id, s.created_at, c.message, c.response, c.timestamp 
                 FROM sessions s
                 LEFT JOIN conversations c ON s.session_id = c.session_id
                 ORDER BY s.created_at DESC, c.timestamp ASC''')
    history = c.fetchall()
    conn.close()
    return history

# Function to create a new session
def create_new_session():
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO sessions (session_id, created_at) VALUES (?, ?)''', 
              (session_id, datetime.now()))
    conn.commit()
    conn.close()
    return session_id

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
    "https://www.workast.com/blog/time-management-and-productivity-in-the-modern-workplace/",
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
   - Start by asking open-ended questions to help the user articulate their problem. Encourage them to share their thoughts and feelings fully.
   - Listen carefully to their responses and ask clarifying questions to deepen your understanding of their situation. Your goal is to make the user feel truly heard and understood.
   - Continue asking reflective questions to guide the user in exploring their thoughts, feelings, and motivations. Avoid jumping to solutions. If the user seems unclear or unsure about their problem, gently help them unpack it with more questions.

2. **Testing Understanding**(Required after giving solutions):
   - After discussing solutions, offer the user an opportunity to practice applying the concepts through hypothetical scenarios. For example, "Imagine you're facing a tight deadline and feeling stressed. How would you approach prioritizing your tasks?"
   - Assess their response and provide constructive feedback. Correct any misunderstandings and guide them on how to improve their approach.

3. **Wrap-Up**(Required):
   - Reassure the user by summarizing their progress. Ask if they'd like to practice another scenario or reflect further on their journey before concluding the session.

Throughout, maintain a supportive and therapeutic tone. If the user does not ask for advice, continue to focus on questions that promote deeper reflection. Only offer suggestions if the user explicitly asks for them. When they seem unsure or off-track, gently steer them back to their challenge without rushing to provide solutions.
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
the most relevant topic.
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

def get_last_n_messages(session_id, n=10):
    """Retrieve the last n messages for a given session."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT message, response, timestamp 
                 FROM conversations 
                 WHERE session_id = ? 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (session_id, n))
    messages = c.fetchall()
    conn.close()
    return list(reversed(messages))  # Return in chronological order

def summarize_history(model, messages):
    """Summarize the chat history using the LLM."""
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """Summarize the following conversation history between a user and an AI mentor. 
                     Focus on the main topics discussed, key insights, and any action items or advice given. 
                     Keep the summary concise but include important context for future reference."""),
        ("human", "{conversation}")
    ])
    
    # Format conversation history into a single string
    conversation = "\n".join([
        f"User: {msg[0]}\nAI: {msg[1]}\n"
        for msg in messages
    ])
    
    # Create and run the summarization chain
    summary_chain = LLMChain(llm=model, prompt=summary_prompt)
    summary = summary_chain.invoke({"conversation": conversation})
    return summary["text"]

def get_context_with_summary(session_id, model, max_messages=10):
    """Get the conversation context with history summary."""
    # Get all messages for the session
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''SELECT message, response, timestamp 
                 FROM conversations 
                 WHERE session_id = ? 
                 ORDER BY timestamp''', (session_id,))
    all_messages = c.fetchall()
    conn.close()
    
    if len(all_messages) <= max_messages:
        return all_messages
    
    # Split into historical and recent messages
    historical_messages = all_messages[:-max_messages]
    recent_messages = all_messages[-max_messages:]
    
    # Summarize historical messages
    history_summary = summarize_history(model, historical_messages)
    
    # Create a summary message tuple
    summary_message = (
        "Previous conversation summary",
        history_summary,
        recent_messages[0][2]  # Use timestamp of first recent message
    )
    
    # Return summary followed by recent messages
    return [summary_message] + recent_messages

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
    # Get context with summary and recent messages
    context = get_context_with_summary(session_id, model)
    
    # Update the conversation history for the chain
    history = get_session_history(session_id)
    
    # Clear existing history to prevent duplication
    history.clear()
    
    for msg in context:
        if msg[0] == "Previous conversation summary":
            history.add_message(AIMessage(content=f"Previous conversation summary: {msg[1]}"))
        else:
            history.add_message(HumanMessage(content=msg[0]))
            history.add_message(AIMessage(content=msg[1]))
    
    # Invoke the chain with the updated history
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    
    return response["answer"]

@app.route('/')
def home():
    session['user_id'] = create_new_session()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    session_id = session.get('user_id', create_new_session())
    response = time_management_mentor(user_input, session_id)
    
    # Save the conversation to the database
    save_conversation(session_id, user_input, response)
    
    return jsonify({'response': response})

@app.route('/history', methods=['GET'])
def get_history():
    history = get_all_conversation_history()
    return jsonify({'history': history})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    new_session_id = create_new_session()
    session['user_id'] = new_session_id
    return jsonify({'status': 'success', 'message': 'Session reset successfully', 'new_session_id': new_session_id})

if __name__ == '__main__':
    setup_database()
    app.run(debug=True)