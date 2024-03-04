import streamlit as st
from streamlit_extras.app_logo import add_logo

from streamlit_extras.colored_header import colored_header
from PIL import Image
from streamlit_card import card

# Head

st.set_page_config(page_title="About", page_icon="ℹ")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "Flip-Flat";
            margin-left: 20px;
            margin-top: 20px;
            font-size: 30px;
            position: relative;
            top: 80px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

colored_header(
    label="Hello! 👋",
    description="",
    color_name="red-70"
)

########## Body ##########

st.write("""
        My name is **Miguel Palos Pou**, and I'm the creator of **Flip-Flat**. I was born in 1996 in Barcelona (Spain) and 
        studied **Engineering** at University of Barcelona.
        Over the last (5) years I have worked in Operations for SaaS B2B startup companies and large Fortune 500 enterprises in Ireland,                     Switzerland, the United States, and Spain.
        I recently pursued a Bootcamp in  **Data Analytics** at Ironhack Barcelona.
        """
        )

st.markdown('<center><img src="https://i.ibb.co/HPhPnf9/P1057937-1.jpg" width=400></center>', unsafe_allow_html=True)
st.write("\n")

st.write("""
        Four years prior, I embarked on a journey as an Operations Analyst, where I implemented agile methodologies and enhanced processes within            the chemical and automotive sectors. Yet, as time progressed, I felt a strong desire for a new beginning. Driven by my love for Data and             storytelling, I enrolled in a 3-month Data Analytics Bootcamp at Ironhack Spain, seeking to pivot my career path.
        """)

st.write("""
        **Flip-flat is the culmination** of the Data Analytics bootcamp at Ironhack and all you see was entirely made in two weeks. 
        A few months later, new features and visuals have been included in order to improve the model and platform.
        """
        )

st.write("""
        **Thank you** for visiting my webpage and I hope this could be the start of a new adventure. 
        You can contact me on my **LinkedIn profile** in case of any doubts.
        """)


card(
    title="LinkedIn",
    text="See my profile",
    image="https://i.ibb.co/HPhPnf9/P1057937-1.jpg",
    url = "https://www.linkedin.com/in/miguelpalospou/"
)   