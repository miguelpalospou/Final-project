import streamlit as st
from PIL import Image
from streamlit_extras.app_logo import add_logo

from streamlit_extras.colored_header import colored_header
from streamlit_card import card

########## Head ##########

st.set_page_config(
    page_title="Flip-Flap",
    page_icon="🏘️",
)


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
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.write("#")
st.markdown(
    '<center><img src="https://i.ibb.co/ZJPw4SP/flip-flap-logo-1.png" style="width:600px;height:400px;"></center>',
    unsafe_allow_html=True
)
colored_header(
    label="WELCOME TO FLIP-FLAP: An AI-based house price predictor",
    description="",
    color_name="yellow-70"
)

########## Body ##########

st.markdown("<h2 style='text-align: center;'>Are you a seller? Know the market trends and increase your profit margin by >5%</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Are you a potential buyer? Understand market price around your desired area and get a fair price for your new house</h2>", unsafe_allow_html=True)
st.image("https://i.ibb.co/G78zzFj/140-DAC2-B-BF83-4-C1-A-A340-68-A550-C549-B3.jpg", use_column_width=True)

st.markdown("<h1 style='text-align: center;'> Real-time data 🏃</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Flip-Flap uses updated data from real estate websites</p>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>More than just a simple predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Tell us your requirements, lay back and trust in us. We will do the rest for you</p>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Reliable </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>We have developed an Artificial Intelligence model that we are constantly improving</p>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Easy to use</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>We know that house search can be a heartache. We want to make things simple</p>", unsafe_allow_html=True)


st.write("\n")
st.write("\n")

########## Cards ##########

st.markdown("<h1 style='text-align: center;'>Check out an example of our service</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>👇👇👇</h1>", unsafe_allow_html=True)

st.image("https://i.ibb.co/tp6h6tC/37198712-9172-46-DF-BDED-D1063318-CB50.jpg", use_column_width=True)




st.markdown(
    '<center><img src="https://i.ibb.co/ZJPw4SP/flip-flap-logo-1.png" style="width:600px;height:400px;"></center>',
    unsafe_allow_html=True
)
