import streamlit as st
from lib.lib_utils import *
from lib.lib_conf import *
from lib.lib_data_ingest import *
def query_history_page():
    session = st.session_state

    st.markdown('Query History')
    username = session.get('login-username')
    query_history = getSessionByUsername(username)

    st.write(f"Total {len(query_history)} queries")
    for i, query in enumerate(query_history):
        keys_list = list(query.data.keys())
        button_label = f"{query.get('selected_users')} :  {query.get('start_date')} ~ {query.get('end_date')}"
        with st.expander(button_label, expanded=False):

            if st.button('query again', key=f'query again {i}'):
                query.setSession(session)
                session['page'] = "results"
                st.experimental_rerun()
            for key in keys_list:
                if (key.startswith('df_') or key.endswith('_df')):
                    continue
                st.markdown(f"<font color='gray' size='2'>{key} : {query.data.get(key)}</font>",
                            unsafe_allow_html=True)

                # st.write(f"{key} : {query.data.get(key)}")