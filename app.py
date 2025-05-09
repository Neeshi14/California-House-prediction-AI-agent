import streamlit as st
from california_house_price_adk_agent import CaliforniaHousePriceAgent
import asyncio

# Streamlit page configuration
st.set_page_config(
    page_title="California House Price Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto"
)

# Title and description in main area
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏠 AI House Price Assistant")
    st.markdown("""
    Welcome! I'm your AI assistant for California house price predictions. 
    I understand natural language, so feel free to ask me anything about house prices in a conversational way!
    """)

# Initialize AI agent
@st.cache_resource
def get_ai_agent():
    return CaliforniaHousePriceAgent()

agent = get_ai_agent()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_message = """👋 Hi! I can help predict house prices in California! Try asking me questions like:

- "What would a 3-bedroom house with 1500 square feet cost?"
- "Can you tell me the price for a house in an area with $50,000 income and 23,000 people?"
- "How much is a 4-bed home that's 2000 sqft in size?"

I'll guide you through any missing details I need! 😊
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Chat input at the top
user_input = st.chat_input("Ask me about house prices in California...")

# Main chat container for history
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Handle new user input
    if user_input:
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    response = asyncio.run(agent.handle_message(None, user_input))
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.markdown(error_msg)

# Helpful information in sidebar
with st.sidebar:
    st.markdown("### 💡 How to Ask")
    st.info("""
    You can ask me in natural language! For example:
    
    🏠 About the house:
    - "3 bedroom house"
    - "1500 square feet"
    - "4 bed, 2000 sqft home"
    
    📊 About the area:
    - "income around $50k"
    - "23000 people live there"
    - "population is 25k"
    
    💬 Or combine everything:
    "How much for a 3-bed house with 1500 sqft in an area with $50k income and 23k people?"
    """)
    
    st.markdown("### 🎯 What I Need")
    st.info("""
    To give you an accurate prediction, I need to know:
    
    1. **House Details**
       - Number of rooms/bedrooms
       - Size in square feet
    
    2. **Area Information**
       - Median income ($)
       - Population
    
    Don't worry about the format - just mention them naturally in your question!
    """)
    
    if st.button("Start Fresh 🔄", use_container_width=True):
        st.session_state.messages = []
        welcome_message = """👋 Hi! I can help predict house prices in California! Try asking me questions like:

- "What would a 3-bedroom house with 1500 square feet cost?"
- "Can you tell me the price for a house in an area with $50,000 income and 23,000 people?"
- "How much is a 4-bed home that's 2000 sqft in size?"

I'll guide you through any missing details I need! 😊
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        st.rerun()