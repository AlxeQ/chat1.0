import os
import pandas as pd
import docx2txt
import tempfile
import pdfplumber
import requests
import streamlit as st

# 从环境变量读取DeepSeek API Key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

def call_deepseek_api(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"调用 DeepSeek API 失败，状态码 {response.status_code}: {response.text}"

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        text = docx2txt.process(tmp.name)
    return text

def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "不支持的文件类型"

def analyze_interview(transcript, outline):
    prompt = f"""
你是一个访谈内容分析助手。请根据以下访谈大纲，对访谈内容进行结构化整理。

访谈大纲：
{outline}

访谈记录：
{transcript}

请生成一个标准 Markdown 表格，包含以下列（列标题请严格按顺序填写）：
| 大纲问题 | 匹配内容摘要 | 覆盖情况（充分 / 部分 / 未覆盖） | 建议补问 |

请确保表格以 `|` 和 `---` 格式写成 Markdown 表格
"""
    return call_deepseek_api(prompt)

def main():
    st.set_page_config(page_title="访谈结构整理工具", layout="wide")
    st.title("📋 访谈结构整理 MVP 工具")

    st.markdown("### 第一步：上传访谈文件（pdf、docx、txt）")
    interview_file = st.file_uploader("上传访谈记录文件：", type=["pdf", "docx", "txt"])

    st.markdown("### 第二步：上传访谈大纲（docx、txt）")
    outline_file = st.file_uploader("上传访谈大纲文件：", type=["docx", "txt"])

    if st.button("🚀 开始分析") and interview_file and outline_file:
        with st.spinner("⏳ 正在提取与分析内容，请稍候..."):
            transcript = extract_text(interview_file)
            outline = extract_text(outline_file)

            result_markdown = analyze_interview(transcript, outline)

        st.markdown("### ✅ 分析结果")
        st.markdown(result_markdown, unsafe_allow_html=True)

        # 导出Excel按钮
        try:
            df_list = pd.read_html(result_markdown, flavor="bs4")
            if df_list:
                df = df_list[0]
                st.download_button(
                    label="📥 下载结果为 Excel",
                    data=df.to_excel(index=False, engine='openpyxl'),
                    file_name="interview_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ 未能识别表格数据，请检查返回的Markdown格式是否正确。")
        except Exception as e:
            st.error(f"❌ 转换结果失败：{e}")

if __name__ == "__main__":
    main()
