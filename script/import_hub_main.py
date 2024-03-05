import streamlit as st
import pandas as pd
import difflib
import re
from fuzzywuzzy import process

from script.utils import load_config

from script.w4h_db_utils import create_w4h_instance, get_existing_databases, populate_tables, populate_subject_table,get_existing_database_server


CONFIG_FILE = 'conf/config.yaml'

def preprocess_string(s: str) -> str:
    """Preprocess the string by converting to lowercase, replacing underscores with spaces,
    tokenizing, and then reconstructing without special characters.
    
    Args:
        s (str): String to preprocess
        
    Returns:
        str: Preprocessed string
    """
    # Convert to lowercase and replace underscores with spaces
    s = s.lower().replace('_', ' ')
    
    # Tokenize the string and remove special characters
    tokens = re.findall(r'\b\w+\b', s)
    
    # Reconstruct the string with spaces
    return ' '.join(tokens)


def find_closest_name(col_names: list, targets: str) -> str:
    """Find the closest column name based on substrings.
    
    Args:
        col_names (list): List of column names to search through
        targets (str): String containing the target column names    
        
    Returns:
        str: Closest column name
    
    """
    score_threshold = 60
    def_choice = col_names[0]  # default
    
    # Preprocess column names
    preprocessed_to_original = {preprocess_string(col): col for col in col_names}
    preprocessed_col_names = list(preprocessed_to_original.keys())
    
    # Extract best match using fuzzywuzzy
    match, score = process.extractOne(preprocess_string(targets), preprocessed_col_names)
    
    return preprocessed_to_original[match] if score > score_threshold else def_choice



def populate_db(df: pd.DataFrame, db_name: str, mappings: dict, config_path: str):
    """Populate the database with the given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the data to be inserted into the database
        db_name (str): Name of the database to insert the data into
        mappings (dict): Dictionary containing the mappings between the CSV columns and the database tables
        config_path (str, optional): Path to the config file. Defaults to 'config.yaml'.
    """
    st.write("Populating database...")
    st.write(mappings)
    populate_tables(df, db_name, mappings, config_path)

def generate_mets_by_calories(df):
    similar_weight = find_closest_name(['None']+list(df.columns), 'weight')
    similar_calories = find_closest_name(df.columns, 'calories')
    print("similar_weight: ", similar_weight)
    if(similar_weight == 'None'):
        df['w4h-mets'] = df[similar_calories] / (70 * 0.25)
    else:
        df['w4h-mets'] = df[similar_calories] / (df[similar_weight] * 0.25)
    return df

def import_page():
    st.subheader("Database Manage")
    if st.button("Edit Database Settings"):
        st.session_state["page"] = "setting"
        st.experimental_rerun()


    """Main function for the streamlit app"""
    # Load the config
    config = load_config(config_file=CONFIG_FILE)
    config_path = CONFIG_FILE


    st.title("W4H Import Hub")

    selected_db = None

    # Choose between existing or new database
    db_selection_options = ["Choose existing W4H database instance", "Create new W4H database instance"]
    database_option = st.radio(
        "Select an option",
        db_selection_options
    )

    # Handling the chosen option
    if database_option == db_selection_options[0]:
        # `get_existing_databases()` retrieves the list of existing databases.
        existing_databases = get_existing_databases()  # This function needs to be implemented.
        
        selected_db = st.selectbox("**Select an existing database**", existing_databases)
        
            
    elif database_option == db_selection_options[1]:

        col1, col2 = st.columns([2,5])
        with col1:
            selected_db_server = st.selectbox("**Select a database server**", get_existing_database_server()).split(' (')[0]
        with col2:
            new_db_name = st.text_input("Enter new w4h database instance name")
        if st.button("Create"):
            # Here, implement logic to create the new database with the name new_db_name.
            print(selected_db_server)
            create_w4h_instance(selected_db_server, new_db_name, config_path)  # This function needs to be implemented.
            st.success(f"Database '{new_db_name}' created!")
            selected_db = "[" + selected_db_server + "] " + new_db_name
            

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # option to populate subject table or feature time series tables
    is_subjects_populated = st.checkbox("Populate subject table?")

    if uploaded_file:
        st.success(f"File `{uploaded_file.name}` uploaded!")
        df = pd.read_csv(uploaded_file)
        st.write("Columns in your CSV:")
        st.write(df.columns)

        # create a mapping between the csv columns and the database tables
        st.subheader("Mapping")
        choices = ["None"] + list(df.columns)
        mappings = {}
        # if subject table is populated, populate subject table
        if is_subjects_populated:
            user_tbl_name = config['mapping']['tables']['user_table']['name']
            with st.expander(f"**Map subject attributes to the W4H `{user_tbl_name}` table attributes**", expanded=True):
                st.write(f"Map your CSV columns to corresponding W4H `{user_tbl_name}` table attributes.")
                for target_attribute in config['mapping']['tables']['user_table']['attributes']:
                    target_attribute_label = ' '.join([label.capitalize() for label in target_attribute['name'].replace('_', ' ').split()])
                    st.write(f"**{target_attribute_label}**")
                    # write the description of the attribute
                    st.write(f'**Description:** {target_attribute["description"]}')
                    # write the data type of the attribute
                    st.write(f'**Data Type:** {target_attribute["type"]}')
                    def_choice = find_closest_name(choices, target_attribute_label)
                    mapped_col = st.selectbox("Select Corresponding Column", choices,
                                            key=target_attribute['name'], index=choices.index(def_choice))
                    mappings[target_attribute['name']] = mapped_col if mapped_col != "None" else None
                    st.markdown("""---""")
            # Once mappings are set, allow the user to populate the database
            if st.button("Populate Database"):
                populate_subject_table(df, selected_db, mappings, config_path)
                st.success(f"Subject table `{user_tbl_name}` populated!")


        # else, populate feature time series tables
        else:
            # Default selections based on column name similarity
            default_timestamp = find_closest_name(df.columns, 'time timestamp date start_time end_time')
            default_user_id = find_closest_name(df.columns, 'user id email patient')

            timestamp_col = st.selectbox("**Select Timestamp Column**", df.columns, index=df.columns.get_loc(default_timestamp))
            user_id_col = st.selectbox("**Select User ID Column**", df.columns, index=df.columns.get_loc(default_user_id))

            # Foldable block for optional mappings
            mappings = {
                config['mapping']['columns']['timestamp']: timestamp_col,
                config["mapping"]['columns']['user_id']: user_id_col,
            }
            table_mappings = {}
            with st.expander("**Map Features to W4H Tables**", expanded=True):
                st.write("Map your CSV columns to corresponding W4H tables.")

                choices = ["None"] + list(df.columns)
                for target_table_name in config['mapping']['tables']['time_series'] + config['mapping']['tables']['geo']:
                    target_table_label = ' '.join([label.capitalize() for label in target_table_name.replace('_', ' ').split()])
                    st.subheader(target_table_label)
                    def_choice = find_closest_name(choices, target_table_label)
                    mapped_col = st.selectbox("Select Corresponding Column", choices,
                                            key=target_table_name, index=choices.index(def_choice))
                    table_mappings[target_table_name] = mapped_col if mapped_col != "None" else None

            # Once mappings are set, allow the user to populate the database
            if st.button("Populate Database"):
                mappings = {**mappings, **table_mappings}

                populate_db(df, selected_db, mappings, config_path)
                st.success("GeoMTS tables populated!")

# if __name__ == "__main__":
#     main()