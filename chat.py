import streamlit as st
from streamlit_chat import message
import pandas as pd
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="centered")
#st.components.v1.html(html, width=None, height=None, scrolling=False)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
local_css("style.css")

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model
@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('bsg_chat.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df=get_dataset()
st.sidebar.title("BSSM")
st.sidebar.info(
    """
    [HomePage] (https://school.busanedu.net/bssm-h/main.do) |\n
    [Instagram] (https://www.instagram.com/bssm.hs/) |\n
    [Facebook] (https://www.facebook.com/BusanSoftwareMEisterHighschool)
    """
)
st.sidebar.title("Contact")
st.sidebar.info(
    """
    tel : 051-971-2153
    """
)
st.header('BSM 챗봇')
tab1, tab2, tab3 = st.tabs(["학교소개", "입학안내", "챗봇"])
with tab1:
    st.header("저희 소마고를 소개합니다.")
    components.html(
    """
    <meta http-equiv="Content-Security-Policy" content="upgrade-insecure-requests">
    <div id="map" style="width:80%;height:300px;">
    <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey=	54bdaa4227094dbb17073d5da58a65a7"></script>
    <script>
    var container = document.getElementById('map');
		var options = {
			center: new kakao.maps.LatLng(35.189260985656, 128.903968747629),
			level: 3
		};
    var map = new kakao.maps.Map(container, options);
    </script>
    </div>
    """,
    height=300,
    )
    st.markdown(
    """
    <div class="school-info">
        <p>주소 : 부산광역시 강서구 가락대로 1393<br>
        전화 : 051-971-2153<br>
        설립 : 1970년 3월 26일<br>
        학생 : 125명 (남 : 89명, 여 : 36명)<br>
        교원 : 33명 (남 : 13명, 여 : 20명)</p>
        </div>
    
    """, unsafe_allow_html=True)
with tab2:
    st.header("입학 안내")
    st.image("https://school.busanedu.net/upload/contents/cntnts/1657675089463_29410551042035296.jpg")
with tab3:
    st.subheader("챗봇에게 무엇이든 물어보세요.")

    #st.subheader('안녕하세요. 소마고 챗봇입니다.')""
    submitted = 0
    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('사용자 : ', '')
        col1, col2 = st.columns(2, gap="small")
        with col1:
            submitted = st.form_submit_button('전송')
        with col2:
            reset_btn = st.form_submit_button('초기화')
    
    if 'generated' not in st.session_state:
        st.session_state['generated']=[]

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if submitted and user_input:
        embedding = model.encode(user_input)
        
        df['distance'] = df['embedding'].map(lambda x:cosine_similarity([embedding], [x]).squeeze())
        answer = df.loc[df['distance'].idxmax()]
        
        st.session_state.past.append(user_input)
        if answer['distance'] >0.7:
            st.session_state.generated.append(answer['챗봇'])
        else:
            st.session_state.generated.append("더 자세한 사항은 051-971-2153으로 문의해주세요")
            
    if reset_btn:
        st.session_state['past'].clear()
        st.session_state['generated'].clear()
        
    for i in range(len(st.session_state['past'])):
    
        st.markdown(
        """
        <div class="right-msg">
            <div class="msg-img"></div>
                <div class="msg-info">
                </div>
                <div class="right-bubble">
                <p>{0}</p>
            </div>
            </div>
            <div class="left-msg">
                <div class="msg-info">
                <div class="msg-info-name">소마고 챗봇</div>
                </div>
                <div class="left-bubble">
                <p>{1}</p>
                </div>
            </div>
        </div>
        """.format(st.session_state['past'][i], st.session_state['generated'][i]),unsafe_allow_html=True,
        )
#for i in range(len(st.session_state['past'])):
 #   message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
  #  if len(st.session_state['generated'])>i:
   #     message(st.session_state['generated'][i], key=str(i) + '_bot')