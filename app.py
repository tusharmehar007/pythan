import streamlit as st

st.title("Hello Tushar")
st.write("Streamlit successfully run ho gaya")

# App Title and Description
st.title("My First Streamlit App")
st.write("This is a simple app to demonstrate the basic functionalities of Streamlit.")

# Interactive Widgets in the Sidebar
st.sidebar.header("User Input Features")

# Text Input
user_name = st.sidebar.text_input("What is your name?", "Tanmay Akolakar")

# Slider
age = st.sidebar.slider("Select your age", 0, 100, 25)

# Selectbox
favorite_color = st.sidebar.selectbox(
    "What is your favorite color?",
    ["Blue", "Red", "Green", "Yellow"]
)

st.write("Hello", user_name)
st.write("Your age:", age)
st.write("Favorite color:", favorite_color)