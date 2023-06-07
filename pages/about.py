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
            content: "Flip-Flap";
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
        My name is **Miguel Palos** and I'm the creator of **Flip-Flap**. was born in 1996 in Barcelona (Spain) and 
        I studied **Engineering** at University of Barcelona. 
        Later on, I pursued a bootcamp in  **Data Analytics** at Ironhack Barcelona.
        """
        )

st.markdown('<center><img src="https://i.ibb.co/HPhPnf9/P1057937-1.jpg" width=400></center>', unsafe_allow_html=True)
st.write("\n")

st.write("""
        Four years ago I started working as a **Operations Engineer** implementing agile ways of working
         and optimizing processes for the chemical and automotive industry. 
        However, in my last years, I realized **I wawnted to make a change** and start something new. 
        My passion to understand data and storytelling brought me to Ironhack Spain, where I coursed a **3-month Bootcamp in Data Analytics**.
        """)

st.write("""
        **Flip-flap is the culmination** of the Data Analytics bootcamp at Ironhack and all you see was entirely made in one week. 
        I can't be more proud of myself for what I have done in such a time.
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