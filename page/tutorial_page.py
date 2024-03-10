import streamlit as st
from lib.lib_utils import load_config,save_config
from lib.lib_conf import *
def tutorial_page():
    st.markdown('Build your config file from here:  ')
    st.markdown('[Tutorial](https://w4h-tutorial.vercel.app/)')
    st.markdown('Then upload here:  ')
    #
    # if page == "Setting up":
    #     with open('markdown/setting_up.md', 'r', encoding='utf-8') as markdown_file:
    #         markdown_text = markdown_file.read()
    # elif page == "How to start":
    #     with open('markdown/how_to_start.md', 'r', encoding='utf-8') as markdown_file:
    #         markdown_text = markdown_file.read()
    # st.markdown(markdown_text, unsafe_allow_html=True)
    # if page == "Setting up":
    config_file = st.file_uploader("Upload config file", type=['yaml', 'example', 'txt'])
    update_config = st.button("Update config")
    if config_file is not None and update_config:
        conf_dir = 'conf'
        if not os.path.exists(conf_dir):
            os.makedirs(conf_dir)
        with open(db_config_path, 'w') as f:
            # write content as string data into the file
            f.write(config_file.getvalue().decode("utf-8"))
        st.success("Update success!")