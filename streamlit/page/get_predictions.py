import streamlit as st
from PIL import Image
import base64
import time

from io import BytesIO

# st.set_page_config(
#         page_title="Predictions",
# )

# Convert Image to Base64 
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

# Convert Base64 to Image
def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)

def run():

    import requests
    url = st.secrets["url"]
    if("query_img" not in st.session_state):
        st.subheader("Error, check your exemplars or the service is broken!")
    else:
        st.subheader(f'Using {len(st.session_state["exemplars_img"])} exemplar(s)')
      

        js = {}
        query =  im_2_b64(st.session_state.predict_img.convert('RGB'))

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
        viz = b64_2_img(r.json()['viz'].split(',')[1])
        st.image(viz)
        st.text('Time taken from request to result: ' + str(end_time-start_time))



def predict_viz():

    img_file = st.file_uploader(label='Upload a file', key="P" , type=['png', 'jpg'])
    if 'predict_img' not in st.session_state:
        st.session_state['predict_path'] = img_file

    else:
        if img_file is None:
        
            img_file = st.session_state.predict_path


    btn_predict=False
    if img_file:
        st.write(img_file.name)
        img = Image.open(img_file)
        st.session_state['predict_img'] = img  
        st.image(img)
        btn_predict=st.button("predict")
        st.session_state['predict_path'] = img_file
    
    if btn_predict:  

        run()