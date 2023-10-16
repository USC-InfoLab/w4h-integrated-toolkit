import sqlite3
import hashlib
import os

# create a ramdom salt
salt = os.urandom(16)

# endcode password with mixxing salt
original_data = "admin".encode('utf-8')
data_with_salt = salt + original_data
hasher = hashlib.sha256()
hasher.update(data_with_salt)
encode_pass = hasher.digest()

#store password
conn = sqlite3.connect('user.db')
cursor = conn.cursor()
cursor.execute('''drop table if exists users''')
cursor.execute('''create table if not exists users (
                    username text primary key,
                    password BLOB,
                    salt BLOB,
                    current_db text)''')
cursor.execute('''insert into users
                    values("admin",?,?,"")''',(encode_pass,salt,))
conn.commit()
conn.close()