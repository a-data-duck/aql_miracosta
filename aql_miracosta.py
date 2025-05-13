import streamlit as st
import json
import requests

# Hide sidebar
st.set_page_config(page_title="MiraCosta College Q&A", page_icon="🎓", initial_sidebar_state="collapsed")

# Custom CSS to hide the sidebar completely
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    .big-font {
        font-size: 24px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .small-italic {
        font-size: 14px;
        font-style: italic;
        color: #666;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for question
if "question" not in st.session_state:
    st.session_state.question = ""

# Set title
st.title("MiraCosta College Q&A")

# Configure API keys (now hidden from sidebar)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_URL = "https://miracosta-docs-h3y3rrq.svc.aped-4627-b74a.pinecone.io"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Missing API keys. Please contact the administrator.")
    st.stop()

# Function to get embedding directly via API
def get_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        st.error(f"OpenAI API error: {response.text}")
        return None
    
    result = response.json()
    return result["data"][0]["embedding"]

# Function for hybrid search - combine keywords and vectors
def hybrid_search(query, base_url, top_k=5):
    # Make sure the URL ends with /query
    if not base_url.endswith("/query"):
        query_url = f"{base_url}/query"
    else:
        query_url = base_url
    
    # Step 1: Generate embedding for vector search
    embedding = get_embedding(query)
    if not embedding:
        return []

    # Step 2: Prepare keywords from query for boosting relevant results
    keywords = query.lower().split()
    # Add specific terms that might be important
    if "wellness" in query.lower() or "health" in query.lower():
        keywords.extend(["timelycare", "wellness", "services", "health"])
    if "program" in query.lower() or "study" in query.lower():
        keywords.extend(["certificate", "program", "course"])
    if "free" in query.lower() or "cost" in query.lower():
        keywords.extend(["tuition", "free", "cost", "financial"])
    
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }
    
    data = {
        "vector": embedding,
        "top_k": top_k * 2,  # Retrieve more than needed for filtering
        "include_metadata": True
    }
    
    try:
        response = requests.post(
            query_url,
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code != 200:
            st.error(f"Pinecone API error: {response.text}")
            return []
        
        # Parse results
        result = response.json()
        matches = result.get("matches", [])
        
        # Now boost relevance scores based on keyword presence
        for match in matches:
            text = match.get("metadata", {}).get("text", "").lower()
            
            # Calculate keyword boost factor
            keyword_matches = sum(1 for keyword in keywords if keyword in text)
            keyword_boost = keyword_matches * 0.1  # Each keyword match adds 0.1 to score
            
            # Apply the boost to the score (keeping it under 1.0)
            match["score"] = min(match["score"] + keyword_boost, 1.0)
        
        # Re-sort matches by adjusted score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k matches
        return matches[:top_k]
            
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# Function to generate answer via OpenAI API
def generate_answer(question, context):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # Enhanced prompt to focus on specific information
    system_prompt = """You are a helpful assistant for MiraCosta College, a California community college.
Answer questions based ONLY on the provided context. If you don't know the answer, say so.
Be specific about services, programs, and resources offered by MiraCosta.
When answering about services like wellness services, ALWAYS mention the specific provider if it appears in the context.
Do NOT generate images or respond to questions unrelated to Calbright College."""
    
    data = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ],
        "temperature": 0.2
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        st.error(f"OpenAI API error: {response.text}")
        return "Sorry, I couldn't generate an answer."
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Main interface
st.write("Ask questions about MiraCosta College's programs, services, and more.")

# Example questions moved to main interface as buttons
st.write("Try an example:")
col1, col2 = st.columns(2)

# Helper function to set question in session state
def set_question(text):
    st.session_state.question = text

with col1:
    if st.button("Who provides wellness services?"):
        set_question("Who provides wellness services at MiraCosta?")
    if st.button("What is tuition at MiraCosta?"):
        set_question("What is tuition at MiraCosta?")
with col2:
    if st.button("What programs are offered?"):
        set_question("What programs does MiraCosta offer?")
    if st.button("How long to complete a program?"):
        set_question("How long does it take to complete a program?")

# Question input
question_input = st.text_input("Or type your own question:", value=st.session_state.question)

# Update session state if user types a question
if question_input != st.session_state.question:
    st.session_state.question = question_input

# Submit button
if st.button("Submit") or (st.session_state.question and not question_input):
    if not st.session_state.question:
        st.warning("Please enter a question or select an example.")
    else:
        try:
            with st.spinner("Searching for information..."):
                # Use hybrid search for better results
                matches = hybrid_search(st.session_state.question, PINECONE_URL)
                if not matches:
                    st.warning("No relevant information found.")
                    st.stop()
                
                # Format context
                context = ""
                sources = []
                
                for i, match in enumerate(matches):
                    metadata = match.get("metadata", {})
                    text = metadata.get("text", "No text available")
                    url = metadata.get("url", "")
                    title = metadata.get("title", "")
                    
                    context += f"\nDocument {i+1}:\n{text}\n"
                    sources.append((title, url))
                
                # Generate answer
                answer = generate_answer(st.session_state.question, context)
                
                # Display answer in larger font (without a heading)
                st.markdown(f'<div class="big-font">{answer}</div>', unsafe_allow_html=True)
                
                # Display sources with smaller, italicized heading
                st.markdown('<div class="small-italic">sources</div>', unsafe_allow_html=True)
                for i, (title, url) in enumerate(sources):
                    st.write(f"{i+1}. {title}")
                    st.write(f"URL: {url}")
                    st.write("---")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again later.")
            st.write(f"Error details: {str(e)}")
