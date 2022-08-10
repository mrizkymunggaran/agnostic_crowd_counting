import os
import streamlit as st
from streamlit_option_menu import option_menu
from page.enroll_exemplars import *
from page.get_predictions import *
from page.get_predictions_noviz import *

# st.set_page_config(
#         page_title="Enroll Exemplars")


with st.sidebar:
    menu = option_menu(
        "Menu",
        [
            "Chicken Enrollment",
            "Predict and Visualization",
            "Predict NON Visualization",
           
        ],
        icons=["images", "eye", "patch-check", "patch-check"],
        menu_icon="cast",
        default_index=0,
    )



if menu == "Chicken Enrollment":
   enroll()

if menu == "Predict and Visualization":
  predict_viz()

if menu == "Predict NON Visualization":
    predict_no_viz()