import streamlit as st
import os
from src.rag_engine import HR_RAG_Engine

st.set_page_config(page_title="Flykite HR Assistant", page_icon="✈️")

st.title("✈️ Flykite Airlines HR Policy Assistant")
st.markdown("Ask questions about company policies, leave, benefits, and more.")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google API Key", type="password", help="Get your key from Google AI Studio")
    
    st.divider()
    
    st.subheader("Data Management")
    # Check if we have data source
    pdf_path = "data/Dataset - Flykite Airlines_ HRP.pdf"
    
    if st.button("Process/Refresh Knowledge Base"):
        if not api_key:
            st.error("Please enter a Google API Key first.")
        elif not os.path.exists(pdf_path):
            st.error(f"Data file not found at: {pdf_path}")
        else:
            with st.spinner("Indexing HR Policy Document... This may take a moment."):
                try:
                    engine = HR_RAG_Engine(api_key)
                    result = engine.ingest_data(pdf_path)
                    st.success(result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    if not api_key:
        st.error("Please enter your Google API Key in the sidebar to continue.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                try:
                    engine = HR_RAG_Engine(api_key)
                    chain = engine.get_qa_chain()
                    response = chain.invoke({"query": prompt})
                    answer = response['result']
                    
                    # Format sources
                    sources = response.get('source_documents', [])
                    if sources:
                        answer += "\n\n**Sources:**\n"
                        for doc in sources:
                            page = doc.metadata.get('page', 'N/A')
                            answer += f"- Page {page}\n"
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
