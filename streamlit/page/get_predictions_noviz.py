import streamlit as st
import base64
import time
from PIL import Image
from io import BytesIO

# st.set_page_config(
#         page_title="Get Predictions",
# )

# Convert Image to Base64 
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

def run():
    import requests
    url = st.secrets["url"]
    if("query_img" not in st.session_state):
        st.subheader("Error, check your exemplars or the service is broken!")
    else:
        st.subheader(f'Using {len(st.session_state["exemplars_img"])} exemplar(s)')
        # st.image(st.session_state.predict_img)
        js = {}
        query =  im_2_b64(st.session_state.predictnon_img.convert('RGB'))

        js['query'] = "data:image/png;base64,"+str(query)[2:]

        exemplars = []
        for ex in st.session_state.exemplars_img:
            ex = im_2_b64(ex)
            exemplars.append("data:image/png;base64,"+str(ex)[2:])
        js['exemplars'] = exemplars
        start_time = time.time()
        r = requests.post(url, json=js)
        end_time = time.time()

        st.text('Count: ' + str(r.json()['count']))
        st.text('Time taken from request to result: ' + str(end_time-start_time))


def predict_no_viz():
    img_file = st.file_uploader(label='Upload a file', key="PN" , type=['png', 'jpg'])
    if 'predictnon_img' not in st.session_state:
        st.session_state['predictnon_path'] = img_file

    else:
        if img_file is None:
            # st.write(img_file)
            img_file = st.session_state.predict_path

    # st.write(img_file)
    # st.write ( 'predictnon_path' not in st.session_state)
    btn_predict=False
    if img_file:
        st.write(img_file.name)
        img = Image.open(img_file)
        st.session_state['predictnon_img'] = img  
        st.image(img)
        btn_predict=st.button("predict")
        st.session_state['predictnon_path'] = img_file

    
    if btn_predict:
            run()

