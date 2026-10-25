
import streamlit as st
import mysql.connector
from mysql.connector import Error

# Database connection function
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="demogrp.mysql.database.azure.com",
            user="bhoomi",
            password="admin@123",
            database="groupbos"
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Insert data into the database
def insert_text_to_database(connection, text_data):
    try:
        cursor = connection.cursor()
        query = "INSERT INTO text_entries (text_data) VALUES (%s)"
        cursor.execute(query, (text_data,))
        connection.commit()
        st.success("Text successfully inserted into the database!")
    except Error as e:
        st.error(f"Error inserting data: {e}")

# Retrieve data from the database
def get_text_from_database(connection):
    try:
        cursor = connection.cursor()
        query = "SELECT id, text_data, created_at FROM text_entries ORDER BY created_at DESC"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except Error as e:
        st.error(f"Error retrieving data: {e}")
        return []

# Streamlit UI
st.title("Send and Retrieve Text from Azure MySQL Database")

# Input text area
user_input = st.text_area("Enter your text here:")

if st.button("Submit"):
    if user_input.strip():
        db_connection = connect_to_database()
        if db_connection:
            insert_text_to_database(db_connection, user_input)
            db_connection.close()
    else:
        st.error("Please enter some text.")

# Display all text from the database
st.header("Previously Entered Texts")
db_connection = connect_to_database()
if db_connection:
    entries = get_text_from_database(db_connection)
    db_connection.close()
    
    if entries:
        for entry in entries:
            st.write(f"**ID:** {entry[0]}")
            st.write(f"**Text:** {entry[1]}")
            st.write(f"**Created At:** {entry[2]}")
            st.write("---")  # Divider line
    else:
        st.info("No entries found in the database.")
