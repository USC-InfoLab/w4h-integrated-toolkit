import streamlit as st
from lib.lib_utils import load_config,save_config
from lib.lib_conf import *
def setting_page():
    st.title("Database Management")

    config = load_config(db_config_path)
    if 'database_number' not in config:
        st.error('key wrong: "database_number" in config')
        return
    if config['database_number'] == 0:
            st.subheader("no saved databases")
    if config['database_number'] != 0:
        for i in range(1, config['database_number'] + 1):
            db_key = f'database{i}'
            if db_key in config:
                db_config = config[db_key]
            with st.expander(f"Database {i} - {db_config['nickname']} ({db_config['dbms']})", expanded=False):
                c1,c2 = st.columns([1,10])
                with c1:
                    is_deleting = st.button('❌', help = 'delete', key=f'delete_{i}')
                    is_saving = st.button('💾', help = 'save', key = f'save_{i}')  # 添加保存按钮


                with c2:

                    nickname = st.text_input("Nickname", db_config['nickname'],key=f'nickname_{i}')
                    dbms = st.selectbox("DBMS", ['postgresql', 'mysql', 'sqlite'],
                                        index=['postgresql', 'mysql', 'sqlite'].index(db_config['dbms']), key=f'selectbox_{i}')
                    host = st.text_input("Host", db_config['host'],key=f'host_{i}')
                    port = st.text_input("Port", db_config['port'],key=f'port_{i}')
                    user = st.text_input("User", db_config['user'],key=f'user_{i}')
                    password = st.text_input("Password", db_config['password'], type="password",key=f'password_{i}')
                    if is_saving:
                        print(f'is_saving: {is_saving}')
                        config[db_key] = {
                            'nickname': nickname,
                            'dbms': dbms,
                            'host': host,
                            'port': port,
                            'user': user,
                            'password': password
                        }
                        save_config(db_config_path,config)
                        st.experimental_rerun()  # 重新运行应用

                    if is_deleting:
                        # 提供删除选项
                        config.pop(db_key)
                        for j in range(i,config['database_number'] + 1):
                            if j == config['database_number']:
                                if i!=j:
                                    config.pop(f'database{j}')
                            else:
                                config[f'database{j}'] = config[f'database{j+1}']
                        config['database_number'] -= 1
                        save_config(db_config_path, config)
                        st.experimental_rerun()  # 重新运行应用



    # 添加新数据库配置
    with st.form("new_db"):
        st.write("Add New Database")
        nickname = st.text_input("Nickname")
        dbms = st.selectbox("DBMS", ['postgresql', 'mysql', 'sqlite'])
        host = st.text_input("Host")
        port = st.text_input("Port")
        user = st.text_input("User")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Add Database")
        if submitted:
            # 添加数据库逻辑
            new_db_key = f'database{config["database_number"] + 1}'
            config[new_db_key] = {
                'nickname': nickname,
                'dbms': dbms,
                'host': host,
                'port': port,
                'user': user,
                'password': password
            }
            config['database_number'] += 1
            save_config(db_config_path ,config)
            st.experimental_rerun()  # 重新运行应用

    # 总体保存按钮
    st.write("")  # 添加一行空白
    if st.button("Save All", ):
        save_config(db_config_path,config)