import streamlit as st

import streamlit as st

def createNav():
    # Using object notation
    isLogin = st.session_state.get("login-state",False)
    loginPage = st.sidebar.button('Log Out' if isLogin else 'Log In',use_container_width=True,type="primary")

    st.sidebar.divider()
    st.sidebar.caption("Import Historical Data / Instantiate a New W4H DB Instance")
    importPage = st.sidebar.button("ImportHub",use_container_width=True,type="secondary")

    st.sidebar.divider()
    st.sidebar.caption("Dashboard / Analyze Subjects Data")

    inputPage = st.sidebar.button("Input Page",use_container_width=True,type="secondary")
    resultPage = st.sidebar.button("Result Page",use_container_width=True,type="secondary")

    st.sidebar.divider()
    st.sidebar.caption("Tutorial")
    tutorial = st.sidebar.button("How to Start",use_container_width=True,type="secondary")

    if (loginPage):
        st.session_state["page"] = "login"
        st.experimental_rerun()
    if (importPage):
        st.session_state["page"] = "import"
        st.experimental_rerun()
    if(inputPage):
        st.session_state["page"] = "input"
        st.experimental_rerun()
    if(resultPage):
        st.session_state["page"] = "result"
        st.experimental_rerun()
    if(tutorial):
        st.session_state["page"] = "tutorial"
        st.experimental_rerun()