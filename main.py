import os
import time
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from crewai import Task, Crew, Process, LLM

from agents.triage_agent import create_triage_agent
from agents.resolver_agent import create_resolver_agent
from agents.action_agent import create_action_agent
from agents.escalation_agent import create_escalation_agent
from tools.sentiment import detect_sentiment

# Load env variables
load_dotenv()
DB_PATH = "database/support_logs.db"

# --- Database Setup ---
def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_name TEXT,
            query TEXT,
            response TEXT,
            sentiment TEXT,
            confidence REAL,
            agent_handled TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(user_name, query, response, sentiment, confidence, agent_handled):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO interactions (timestamp, user_name, query, response, sentiment, confidence, agent_handled)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, query, response, sentiment, confidence, agent_handled))
    conn.commit()
    conn.close()

# --- Agent Orchestration ---
def run_support_crew(query: str, username: str, api_key: str, model_name: str = "groq/llama-3.1-8b-instant") -> dict:
    if not api_key:
        return {"response": "Error: Please provide a valid GROQ_API_KEY in the sidebar or .env file.", "agent": "System"}
    
    # Run standalone sentiment to log accurate stats
    sent_res = detect_sentiment(query)
    sentiment = sent_res["sentiment"]
    confidence = sent_res["confidence"]

    try:
        llm = LLM(
            model=model_name,
            temperature=0.3,
            max_tokens=512,
            api_key=api_key,
            max_retries=5
        )
        
        triage_agent = create_triage_agent(llm)
        resolver_agent = create_resolver_agent(llm)
        action_agent = create_action_agent(llm)
        escalation_agent = create_escalation_agent(llm)

        triage_task = Task(
            description=f"Analyze query: '{query}'. Provide intent, sentiment, confidence, and strongly recommend Resolver, Action, or Escalation.",
            agent=triage_agent,
            expected_output="Routing decision and sentiment analysis."
        )
        resolver_task = Task(
            description=f"If Triage recommended Resolver, answer '{query}' using FAQ tool. Else bypass. Reply conversationally.",
            agent=resolver_agent,
            expected_output="Final user answer from knowledge base or bypass."
        )
        action_task = Task(
            description=f"If Triage recommended Action, execute password reset or ticket for user '{username}'. Provide confirmation.",
            agent=action_agent,
            expected_output="Action confirmation or bypass."
        )
        escalation_task = Task(
            description=f"If Triage recommended Escalation, create an escalated ticket for user '{username}' about '{query}'. Provide handoff message.",
            agent=escalation_agent,
            expected_output="Escalation ticket confirmation or bypass."
        )

        crew = Crew(
            agents=[triage_agent, resolver_agent, action_agent, escalation_agent],
            tasks=[triage_task, resolver_task, action_task, escalation_task],
            process=Process.sequential,
            verbose=False
        )

        # Robust Retry Implementation with Stronger Exponential Backoff
        result_str = ""
        import random
        for attempt in range(8):
            try:
                result_str = str(crew.kickoff())
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "429" in err_msg or "too many requests" in err_msg:
                    wait = (2 ** attempt) * 3 + random.uniform(1, 3)
                    st.toast(f"Rate limit hit. Retrying in {int(wait)}s... (Attempt {attempt+1}/8)", icon="⏳")
                    time.sleep(wait)
                    continue
                else:
                    raise e
        
        if not result_str:
            return {"response": "I'm experiencing high load right now. Please try again in 30 seconds.", "agent": "System", "sentiment": sentiment, "confidence": confidence}

        # Simple heuristic... (keep existing)...
        if "Ticket" in result_str and sentiment == "negative":
            handled_by = "Escalation Agent"
        elif "password" in result_str.lower() or "reset" in result_str.lower():
            handled_by = "Action Agent"
        elif "escalated" in result_str.lower():
            handled_by = "Escalation Agent"
        else:
            handled_by = "Resolver Agent"

        return {"response": result_str, "agent": handled_by, "sentiment": sentiment, "confidence": confidence}

    except Exception as e:
        return {"response": f"System Error: {str(e)}", "agent": "System", "sentiment": sentiment, "confidence": confidence}


# --- Streamlit UI ---
st.set_page_config(page_title="Capgemini AI Support", page_icon="💡", layout="wide")
init_db()

# Custom CSS for Branding
st.markdown("""
<style>
    .capgemini-blue { color: #0070ad !important; }
    .stButton>button { border-radius: 20px; border: 2px solid #0070ad; color: #0070ad; }
    .stButton>button:hover { background-color: #0070ad; color: white; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9d/Capgemini_201x_logo.svg", width=150)
    st.markdown("### Settings")
    st.session_state.username = st.text_input("User Name", value="Test User")
    st.session_state.api_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    
    st.markdown("---")
    st.markdown("### Model Selection")
    st.session_state.model_name = st.selectbox(
        "Choose AI Model",
        options=["groq/llama-3.1-8b-instant", "groq/gemma2-9b-it"],
        index=0,
        help="Switch model if you hit rate limits frequently."
    )
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the Capgemini AI Support Agent. How can I assist you today?"}
    ]

# Layout Tabs
tab1, tab2 = st.tabs(["💬 Main Chat Interface", "📊 Live Monitoring Dashboard"])

# TAB 1: Chat Interface
with tab1:
    st.markdown("<h1 class='capgemini-blue'>🤖 Intelligent Support Agent</h1>", unsafe_allow_html=True)
    
    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "meta" in msg:
                st.caption(f"*Handled by: {msg['meta']['agent']} | Confidence: {msg['meta']['confidence']}% | Sentiment: {msg['meta']['sentiment']}*")

    # Chat Input
    if query := st.chat_input("Describe your issue..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("AI Agents analyzing and resolving..."):
                start_time = time.time()
                result = run_support_crew(
                    query, 
                    st.session_state.username, 
                    st.session_state.api_key,
                    st.session_state.model_name
                )
                resp = result.get("response", "Error occurred.")
                agent = result.get("agent", "System")
                conf = int(result.get("confidence", 0.0) * 100)
                sent = result.get("sentiment", "unknown").capitalize()
                
                # Log to DB
                if agent != "System":
                    log_interaction(st.session_state.username, query, resp, sent, conf, agent)
                
                st.write(resp)
                st.caption(f"*Handled by: {agent} | Confidence: {conf}% | Sentiment: {sent}*")
                
                # Save to memory
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": resp,
                    "meta": {"agent": agent, "confidence": conf, "sentiment": sent}
                })

# TAB 2: Monitoring Dashboard
with tab2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9d/Capgemini_201x_logo.svg", width=150)
    st.markdown("<h2 class='capgemini-blue'>Capgemini Live Monitoring Dashboard</h2>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data"):
        st.rerun()
        
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    conn.close()

    if not df.empty:
        total_queries = len(df)
        escalated = len(df[df['agent_handled'] == 'Escalation Agent'])
        resolution_rate = ((total_queries - escalated) / total_queries) * 100
        avg_confidence = df['confidence'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", total_queries)
        col2.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        col3.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        col4.metric("Escalations", escalated)

        st.markdown("### Sentiment Breakdown")
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts, color="#0070ad")

        st.markdown("### Recent Interactions (Last 10)")
        st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)
    else:
        st.info("No logs available yet. Start chatting to populate the dashboard! 🚀")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        init_db()
        print("--- Running End-to-End CLI Tests ---")
        queries = [
            "Please reset my password for testuser",
            "My screen went blank and I am very angry!",
            "How do I track my order?"
        ]
        api_key = os.getenv("GROQ_API_KEY", "no-key")
        for q in queries:
            print(f"\n[QUERY]: {q}")
            res = run_support_crew(q, "CLI_Tester", api_key)
            print(f"[RESULT]: {res['response']}")
