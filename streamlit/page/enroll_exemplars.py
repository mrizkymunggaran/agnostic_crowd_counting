import os
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import uuid
from streamlit_option_menu import option_menu

# st.set_page_config(
#         page_title="Enroll Exemplars")


def enroll():

  st.set_option('deprecation.showfileUploaderEncoding', False)


  if 'id' not in st.session_state:
    st.session_state['id'] = uuid.uuid1()
    os.mkdir(str(st.session_state.id))

  st.header("Class Agnostic Counting Demo")
  st.text("Refresh the tab if you want to change image")



  img_file = st.file_uploader(label='Upload a file', key="U", type=['png', 'jpg'])
  if 'query_img' not in st.session_state:
    # img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    st.session_state['query_path'] = img_file
  else:
    img_file = st.session_state.query_path

  # img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
  # st.session_state['query'] = img_file
  if img_file is not None:
    st.write(img_file.name)
    num_exemplar=3 #default
    if "num_exemplar" in st.session_state:
      num_exemplar=st.session_state['num_exemplar']


    exemplars = st.slider(label='Amount of Exemplars', min_value=1, max_value=10, value=num_exemplar)


    st.session_state['num_exemplar']=exemplars

    # if exemplars != num_exemplar:
    st.session_state['exemplars_img'] = []
    for i in range(exemplars):
      if img_file:
          img = Image.open(img_file)
          st.session_state['query_img'] = img

          st.subheader(f'Exemplar {i+1}')
          cropped_img = st_cropper(img, realtime_update=True, key=f'ex{i}',)
          st.write(f"Preview for Exemplar {i+1}")
          _ = cropped_img.thumbnail((150,150))
          st.image(cropped_img)
          st.session_state['exemplars_img'].append(cropped_img)
      else:
        st.subheader('Image not picked!')
        break

