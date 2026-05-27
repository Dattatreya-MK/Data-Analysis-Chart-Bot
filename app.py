import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Load Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Check API key
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Remember previous conversations."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain
chain = prompt | llm

# Memory Store
if "store" not in st.session_state:
    st.session_state.store = {}

store = st.session_state.store

# Session History Function
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Add memory wrapper
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Streamlit UI
st.title("🤖 Chatbot with Memory using LangChain + Groq")

# Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# Chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
user_input = st.chat_input("Type your message here...")

if user_input:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Get bot response
    response = chain_with_memory.invoke(
        {"input": user_input},
        config={
            "configurable": {
                "session_id": st.session_state.session_id
            }
        }
    )

    bot_reply = response.content

    # Store bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    # Display bot response
    with st.chat_message("assistant"):
        st.write(bot_reply)

# Clear Chat
if st.button("Clear Conversation"):
    st.session_state.messages = []
    store.clear()
    st.rerun()