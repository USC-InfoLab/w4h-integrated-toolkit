import streamlit as st
from lib.lib_utils import *
from lib.lib_conf import *
from lib.lib_data_ingest import *
import hashlib
def login_page():
    st.title("User login")

    username = st.text_input("username")
    password = st.text_input("password", type="password")

    if 'login-state' in st.session_state.keys():
        del st.session_state['login-state']

    if st.button("login"):
        conn = sqlite3.connect('user.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''select password,salt from users where username = ?''',(username,))
            row = cursor.fetchone()
            if row is None:
                st.error("user not exist!")
                conn.close()
                return
            hasher = hashlib.sha256()
            hasher.update(row[1] + password.encode('utf-8'))
            encodePwd = hasher.digest()
            if (row[0] == encodePwd):
                st.session_state["login-state"] = True
                st.session_state["login-username"] = username
                st.session_state["page"] = "input"
                st.experimental_rerun()
            else:
                st.error("username or password is wrong")
        except Exception as err:
            st.error(err)
            st.error("something wrong in the server")
        conn.close()