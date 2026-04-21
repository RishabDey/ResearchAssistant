import streamlit as st
import time
from agents import (build_search_agent,build_reader_agent,writer_chain,critic_chain)

# ── Page config
st.set_page_config(
    page_title="Research Assistant: A Source-Aware Multi-Agent Research Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# ── Sidebar  
with st.sidebar:
    st.markdown("#### Research Assistant")
    st.markdown(
        "<span style='font-size:0.82rem; color:#475569;'>Multi-agent pipeline · LangChain + Mistral</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div style='font-size:0.95rem; line-height:1.9; color:#64748b;'>
        <b style='color:#94a3b8;'>Pipeline stages</b><br>
        1 · <b style='color:#94a3b8;'>Search</b>: discovers relevant sources<br>
        2 · <b style='color:#94a3b8;'>Reader</b>: extracts deep content<br>
        3 · <b style='color:#94a3b8;'>Writer</b>: structures a report<br>
        4 · <b style='color:#94a3b8;'>Evaluater</b>: evaluates and scores
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

# ── Header  
st.markdown(
    """
    <div style='padding: 1.8rem 0 1.2rem;'>
        <h2 style='margin:0; font-size:1.7rem;'>Research Assistant</h2>
        <p style='color:#475569; font-size:0.9rem; margin:0.4rem 0 0;'>
            Enter a topic and let the agent pipeline research, write, and evaluate a report for you.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input row  
col_input, col_btn = st.columns([4, 1])

with col_input:
    topic = st.text_input(
        "Topic to Research",
        placeholder="e.g. Latest news related to job market in Germany",
    )

with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True) 
    run_button = st.button("Create Report", type="primary", use_container_width=True)


# ── Session state  
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}
if "pipeline_topic" not in st.session_state:
    st.session_state.pipeline_topic = ""
if "is_running" not in st.session_state:
    st.session_state.is_running = False

if run_button:
    if not topic.strip():
        st.error("Please enter a research topic before running.")
    else:
        st.session_state.pipeline_results = {}
        st.session_state.pipeline_topic = topic.strip()
        st.session_state.is_running = True
        st.rerun()


# ── Pipeline execution  
if st.session_state.is_running:
    results = {}
    t = st.session_state.pipeline_topic

    with st.status("Searching the web for sources…", expanded=True) as status:
        st.write("Running Search Agent…")
        search_agent = build_search_agent()
        search_output = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {t}")]
        })
        results["search"] = search_output["messages"][-1].content
        st.session_state.pipeline_results["search"] = results["search"]
        status.update(label="Search complete", state="complete")

    with st.status("Reading and extracting content…", expanded=True) as status:
        st.write("Selecting the best source and scraping content…")
        reader_agent = build_reader_agent()
        reader_output = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on these search results about '{t}', select the best URL and scrape detailed content.\n\n"
                f"Search Results:\n{results['search'][:1000]}"
            )]
        })
        results["reader"] = reader_output["messages"][-1].content
        st.session_state.pipeline_results["reader"] = results["reader"]
        status.update(label="Content extracted", state="complete")

    with st.status("Drafting the report…", expanded=True) as status:
        combined_research = (
            f"WEB SEARCH RESULTS:\n{results['search']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
        )
        report = writer_chain.invoke({"topic": t, "research": combined_research})
        results["report"] = report
        st.session_state.pipeline_results["report"] = report
        status.update(label="Report drafted", state="complete")

    with st.status("Critic reviewing the report…", expanded=True) as status:
        feedback = critic_chain.invoke({"report": report})
        results["feedback"] = feedback
        st.session_state.pipeline_results["feedback"] = feedback
        status.update(label="Review complete", state="complete")

    st.session_state.is_running = False
    st.rerun()


# ── Results display  
results = st.session_state.pipeline_results

if results:
    st.divider()

    # Raw outputs 
    with st.expander("Click to see the Raw agent outputs", expanded=False):
        tab_search, tab_reader = st.tabs(["Search Agent", "Reader Agent"])
        with tab_search:
            st.text_area(
                "search_output", results.get("search", ""),
                height=260, label_visibility="collapsed"
            )
        with tab_reader:
            st.text_area(
                "reader_output", results.get("reader", ""),
                height=260, label_visibility="collapsed"
            )

    # Final report
    if "report" in results:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">📄 Report</div>', unsafe_allow_html=True)
        st.markdown(results["report"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
            label="Download as Markdown",
            data=results["report"],
            file_name=f"report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Evaluate report
    if "feedback" in results:
        st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">🔎 Evaluation Report </div>', unsafe_allow_html=True)
        st.markdown(results["feedback"])
        st.markdown('</div>', unsafe_allow_html=True)

    topic_display = st.session_state.get("pipeline_topic", "")
    st.caption(f"Topic: {topic_display} · Report created at: {time.strftime('%H:%M, %d %b %Y')}")


# ── Footer  
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1e293b; font-size:0.8rem;'>"
    "Research Assistant · LangChain + Mistral AI"
    "</p>",
    unsafe_allow_html=True,
)