import os
import tempfile
import pandas as pd
import streamlit as st
from docx import Document
import pdfplumber
import requests
import json
import time

# 设置页面标题和布局
st.set_page_config(page_title="访谈智能整理器", layout="wide")

# 标题和介绍
st.title("📝 访谈智能整理器 (Smart Interview Sorter)")
st.markdown("""
    上传您的访谈记录和大纲，工具将自动整理内容并识别覆盖情况。
    **使用DeepSeek API进行内容分析**
    """)

# 初始化会话状态
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'interview_text' not in st.session_state:
    st.session_state.interview_text = ""
if 'outline_questions' not in st.session_state:
    st.session_state.outline_questions = []

# 文件上传和处理函数
def extract_text_from_file(file):
    """根据文件类型提取文本"""
    text = ""
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    
    try:
        if file.name.endswith('.pdf'):
            with pdfplumber.open(tmp_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file.name.endswith('.docx'):
            doc = Document(tmp_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
        elif file.name.endswith('.txt'):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        st.error(f"文件处理错误: {str(e)}")
    finally:
        os.unlink(tmp_path)
    
    return text

def extract_questions_from_outline(text):
    """从大纲文本中提取问题"""
    # 简单按行分割并过滤空行
    questions = [line.strip() for line in text.split('\n') if line.strip()]
    return questions

def analyze_with_deepseek(interview_text, questions):
    """使用DeepSeek API分析访谈内容"""
    # 在侧边栏输入的API密钥
    api_key = st.session_state.get('deepseek_api_key', '')
    
    if not api_key:
        st.error("请先在侧边栏输入DeepSeek API密钥")
        return None
    
    # 构造提示词
    prompt = f"""
    你是一个专业的访谈内容分析助手。请根据以下访谈记录内容，将回答对应到以下问题中，并评估覆盖情况:

    === 访谈记录 ===
    {interview_text}

    === 问题列表 ===
    {questions}

    请按以下格式返回结果:
    问题 | 匹配内容摘要 | 覆盖情况(充分/部分/未覆盖) | 建议补问内容

    要求:
    1. 每个问题一行，字段之间用 | 分隔
    2. 匹配内容摘要要简洁，不超过50字
    3. 覆盖情况评估标准:
       - 充分: 有直接明确的回答
       - 部分: 有相关但不完全的回答
       - 未覆盖: 完全没有相关内容
    4. 建议补问内容要具体可行
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的访谈内容分析助手。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            st.error(f"DeepSeek API错误: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"API请求异常: {str(e)}")
        return None

def parse_response(response):
    """解析API返回的结果"""
    lines = [line.strip() for line in response.split('\n') if line.strip() and '|' in line]
    data = []
    
    for line in lines:
        parts = [part.strip() for part in line.split('|')]
        if len(parts) >= 4:
            data.append({
                "大纲问题": parts[0],
                "匹配内容摘要": parts[1],
                "覆盖情况": parts[2],
                "建议补问": parts[3]
            })
        elif len(parts) == 3:
            data.append({
                "大纲问题": parts[0],
                "匹配内容摘要": parts[1],
                "覆盖情况": parts[2],
                "建议补问": "无"
            })
    
    return pd.DataFrame(data)

# 主界面
if not st.session_state.processed:
    with st.form("upload_form"):
        st.subheader("1. 上传文件")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**上传访谈记录**")
            interview_file = st.file_uploader(
                "支持 .pdf 或 .docx 格式",
                type=['pdf', 'docx'],
                key="interview_uploader",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**上传访谈大纲**")
            outline_file = st.file_uploader(
                "支持 .docx 或 .txt 格式",
                type=['docx', 'txt'],
                key="outline_uploader",
                label_visibility="collapsed"
            )
        
        submitted = st.form_submit_button("开始整理")
        
        if submitted:
            if not interview_file or not outline_file:
                st.warning("请上传访谈记录和大纲文件")
            elif not st.session_state.get('deepseek_api_key'):
                st.warning("请先在侧边栏输入DeepSeek API密钥")
            else:
                with st.spinner("处理文件中..."):
                    # 提取访谈文本
                    st.session_state.interview_text = extract_text_from_file(interview_file)
                    
                    # 提取大纲问题
                    outline_text = extract_text_from_file(outline_file)
                    st.session_state.outline_questions = extract_questions_from_outline(outline_text)
                    
                    # 使用DeepSeek分析内容
                    st.info("正在使用DeepSeek API分析访谈内容，这可能需要一些时间...")
                    response = analyze_with_deepseek(
                        st.session_state.interview_text, 
                        st.session_state.outline_questions
                    )
                    
                    if response:
                        st.session_state.results = parse_response(response)
                        st.session_state.processed = True
                        st.rerun()

# 结果显示页面
if st.session_state.processed and st.session_state.results is not None:
    st.subheader("📊 分析结果")
    
    # 显示统计信息
    coverage_counts = st.session_state.results['覆盖情况'].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("总问题数", len(st.session_state.results))
    col2.metric("充分覆盖", coverage_counts.get("充分", 0))
    col3.metric("未覆盖", coverage_counts.get("未覆盖", 0))
    
    # 显示数据表格
    st.dataframe(
        st.session_state.results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "大纲问题": "大纲问题",
            "匹配内容摘要": st.column_config.TextColumn(
                "匹配内容摘要",
                width="medium"
            ),
            "覆盖情况": st.column_config.SelectboxColumn(
                "覆盖情况",
                options=["充分", "部分", "未覆盖"],
                default="未覆盖"
            ),
            "建议补问": st.column_config.TextColumn(
                "建议补问",
                width="medium"
            )
        }
    )
    
    # 添加下载按钮
    excel_file = pd.ExcelWriter("interview_results.xlsx", engine='openpyxl')
    st.session_state.results.to_excel(excel_file, index=False)
    excel_file.close()
    
    with open("interview_results.xlsx", "rb") as file:
        st.download_button(
            label="下载Excel文件",
            data=file,
            file_name="interview_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # 显示原始内容和大纲问题
    with st.expander("查看原始访谈内容"):
        st.text_area("访谈记录", st.session_state.interview_text, height=300)
    
    with st.expander("查看大纲问题列表"):
        st.write(st.session_state.outline_questions)
    
    # 重新开始按钮
    if st.button("重新开始"):
        st.session_state.processed = False
        st.session_state.results = None
        st.session_state.interview_text = ""
        st.session_state.outline_questions = []
        st.rerun()

# 侧边栏信息
with st.sidebar:
    st.markdown("## DeepSeek API 设置")
    api_key = st.text_input(
        "DeepSeek API密钥",
        type="password",
        key="deepseek_api_key",
        help="请从DeepSeek平台获取API密钥"
    )
    
    st.markdown("## 使用说明")
    st.markdown("""
    1. 输入DeepSeek API密钥
    2. 上传访谈记录(PDF/DOCX)和访谈大纲(DOCX/TXT)
    3. 点击"开始整理"按钮
    4. 查看分析结果并下载Excel文件
    """)
    
    st.markdown("## 注意事项")
    st.markdown("""
    - 访谈记录应包含完整的对话内容
    - 大纲文件应每行一个主要问题
    - 分析结果基于AI模型，可能需要人工复核
    - 请妥善保管您的API密钥
    """)
    
    st.markdown("---")
    st.markdown("**版本**: v1.1.0")
    st.markdown("**开发者**: Your Name")

# 隐藏Streamlit默认样式
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)