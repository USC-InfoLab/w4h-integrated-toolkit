import streamlit as st

import streamlit as st

def createNav():
    # Using object notation
    isLogin = st.session_state.get("login-state",False)
    loginPage = st.sidebar.button('log out' if isLogin else 'log in',use_container_width=True,type="primary")

    st.sidebar.divider()
    st.sidebar.caption("Set up your patient DB")
    importPage = st.sidebar.button("import page",use_container_width=True,type="secondary")

    st.sidebar.divider()
    st.sidebar.caption("Manage your patients")

    inputPage = st.sidebar.button("input page",use_container_width=True,type="secondary")
    resultPage = st.sidebar.button("result page",use_container_width=True,type="secondary")

    st.sidebar.divider()
    st.sidebar.caption("tutorial")
    tutorial = st.sidebar.button("how to start",use_container_width=True,type="secondary")

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