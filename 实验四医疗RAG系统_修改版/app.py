import streamlit as st
import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache' 
# import evaluation
from chroma_utils import get_chroma_client
from config import (
    EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, RERANKING_ENABLED, RERANKING_MODEL_NAME,
    EVAL_DATA_PATH, RECALL_TOP_K, EMBEDDING_DIM, MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE, INDEX_PARAMS,
    SEARCH_PARAMS, TOP_K, id_to_doc_map
)
# Import functions and config from other modules
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K, GENERATION_TEMPERATURE,
    MAX_ARTICLES_TO_INDEX, CHROMA_PERSIST_DIR, COLLECTION_NAME, RERANKING_ENABLED, RERANKING_MODEL_NAME, RERANKING_NUM, NUM_CANDIDATES,
    id_to_doc_map # Import the global map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model, load_reranking_model
# Import the new ChromaDB functions
from chroma_utils import get_chroma_client, setup_chroma_collection, index_data_if_needed, search_similar_documents
from rag_core import generate_answer, no_doc_generate_answer



# 新增：初始化对话历史存储
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# # 在侧边栏添加评估选项
# st.sidebar.header("评估功能")
# # 修改评估按钮的回调逻辑
# if st.sidebar.button("运行召回评估"):
#     st.write(f"当前评估数据路径：{EVAL_DATA_PATH}")  # 新增路径显示
#     # 使用已初始化的全局客户端实例
#     if get_milvus_client() and EMBEDDING_MODEL_NAME:  # Changed to use get_milvus_client()
#         with st.spinner("正在执行召回评估..."):
#             from evaluation import run_evaluation
#             metrics = run_evaluation(get_milvus_client(), load_embedding_model(EMBEDDING_MODEL_NAME))  # Changed here
            
#             st.subheader("评估结果")
#             # 动态生成指标显示
#             if metrics:
#                 chart_data = {}
#                 for k in RECALL_TOP_K:
#                     key = f"recall@{k}"
#                     if key in metrics:
#                         st.metric(label=key, value=f"{metrics[key]:.2%}")
#                         chart_data[key] = metrics[key]
                
#                 # 仅在有数据时显示图表
#                 if chart_data:
#                     st.bar_chart(chart_data)
#                 else:
#                     st.warning("未找到有效评估指标")
#             else:
#                 st.error("评估失败，未生成任何指标")
                
#             # 删除以下冗余的可视化代码块
#             # st.bar_chart({
#             #     "Recall@10": metrics["recall@10"],
#             #     "Recall@20": metrics["recall@20"], 
#             #     "Recall@50": metrics["recall@50"]
#             # })
#     else:
#         st.error("无法运行评估，请先初始化系统")






# --- Streamlit UI 设置 ---
# st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 (ChromaDB)")
st.markdown(f"使用 ChromaDB, `{EMBEDDING_MODEL_NAME}`, 和 `{GENERATION_MODEL_NAME}`。")

# --- 初始化与缓存 ---
# 获取 ChromaDB 客户端 (如果未缓存则初始化)
chroma_client = get_chroma_client()

if chroma_client:
    # 设置 collection (如果未缓存则创建/加载索引)
    collection_is_ready = setup_chroma_collection(chroma_client)

    # 加载模型 (缓存)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)
    reranking_model = load_reranking_model(RERANKING_MODEL_NAME) if RERANKING_ENABLED else None  # 加载重排序模型

    # # --- 新增：设备统一配置 ---
    # import torch
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    # # 移动模型到目标设备
    # embedding_model = embedding_model.to(device)
    # generation_model = generation_model.to(device)
    # if reranking_model:
    #     reranking_model = reranking_model.to(device)

    # 检查所有组件是否成功加载
    models_loaded = embedding_model and generation_model and tokenizer

    if collection_is_ready and models_loaded:
        # 加载数据 (未缓存)
        pubmed_data = load_data(DATA_FILE)

        # 如果需要则索引数据 (这会填充 id_to_doc_map)
        if pubmed_data:
            indexing_successful = index_data_if_needed(chroma_client, pubmed_data, embedding_model)
        else:
            st.warning(f"无法从 {DATA_FILE} 加载数据。跳过索引。")
            indexing_successful = False # 如果没有数据，则视为不成功

        st.divider()

        # --- RAG 交互部分 ---
        if not indexing_successful and not id_to_doc_map:
             st.error("数据索引失败或不完整，且没有文档映射。RAG 功能已禁用。")
        else:
            # 新增：显示对话历史
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # 替换原单次输入为聊天输入框
            query = st.chat_input("请提出关于已索引医疗文章的问题:")
            
            if query:
                # 记录用户当前提问到对话历史
                with st.chat_message("user"):
                    st.markdown(query)
                st.session_state.chat_history.append({"role": "user", "content": query})

                start_time = time.time()

                history_text = " ".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in st.session_state.chat_history[:-1]  # 排除当前刚添加的用户提问（避免重复）
                    ])

                search_query = f"{history_text} 用户当前问题: {query}"

                # st.write(id_to_doc_map)
                # 1. 搜索 ChromaDB
                with st.spinner("正在搜索相关文档..."):

                    retrieved_ids, distances = search_similar_documents(chroma_client, search_query, embedding_model)
                    # retrieved_ids, distances = search_similar_documents(milvus_client, search_query, reranking_model)

                if not retrieved_ids:
                    st.warning("在数据库中找不到相关文档。")
                else:
                    # 2. 从映射中检索上下文
                    retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]

                    if not retrieved_docs:
                         st.error("检索到的 ID 无法映射到加载的文档。请检查映射逻辑。")
                    else:
                        # 3. 新增：重排序文档（如果启用）
                        if RERANKING_ENABLED and reranking_model:
                            with st.spinner("正在重排序文档..."):
                                # 准备查询-文档对（格式：[(query, doc_content), ...]）
                                sentence_pairs = [(query, doc['content']) for doc in retrieved_docs]
                                # 预测相关性分数（分数越高越相关）
                                scores = reranking_model.predict(sentence_pairs)
                                # 按分数降序排序文档
                                reranked = sorted(zip(retrieved_docs, scores, retrieved_ids), key=lambda x: x[1], reverse=True)
                                retrieved_docs = [item[0] for item in reranked]  # 重新排序后的文档列表
                                retrieved_ids = [item[2] for item in reranked]    # 同步更新ID顺序（可选）
                                distances = [item[1] for item in reranked]       # 用重排序分数替换原距离

                        # 4. 显示检索结果
                        st.subheader("检索到的上下文文档:")
                        for i, doc in enumerate(retrieved_docs):
                            # 显示重排序后的分数（如果启用）
                            if (i >= RERANKING_NUM): break
                            score_str = f", 重排序分数: {distances[i]:.4f}" if RERANKING_ENABLED else ""
                            with st.expander(f"文档 {i+1} (ID: {retrieved_ids[i]}{score_str}) - {doc['title'][:60]}"):
                                st.write(f"**标题:** {doc['title']}")
                                st.write(f"**摘要:** {doc['abstract']}") # 假设 'abstract' 存储的是文本块

                        st.divider()

                        # 3. 生成答案
                        st.subheader("生成的答案:")
                        with st.spinner("正在根据上下文生成答案..."):
                            # 修改：新增 chat_history 参数传递
                            answer = generate_answer(query, retrieved_docs, generation_model, tokenizer, st.session_state.chat_history)
                            response = answer
                        
                        # with st.spinner("正在根据上下文生成答案..."):
                        #     # 传递新增的多样性控制参数
                        #     answer_candidates = generate_answer(
                        #         query, 
                        #         retrieved_docs, 
                        #         generation_model, 
                        #         tokenizer, 
                        #         st.session_state.chat_history,
                        #         num_candidates=NUM_CANDIDATES,
                        #         temperature=GENERATION_TEMPERATURE
                        #     )
                        #     # 选择排序后的第一个作为主回答（也可随机选择）
                        #     response = answer_candidates[0]


                        # 记录助理回答到对话历史并显示
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        with st.spinner("正在生成无上下文回答..."):
                        # 传入空文档列表表示无上下文
                            answer = no_doc_generate_answer(query, generation_model, tokenizer, st.session_state.chat_history)
                        # 记录并显示对照回答
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                        
                        #  # 新增：显示所有候选答案（可选）
                        # with st.expander(f"查看全部 {NUM_CANDIDATES} 个候选答案"):
                        #     for i, candidate in enumerate(answer_candidates):
                        #         st.markdown(f"**候选 {i+1}:** {candidate}")

                        end_time = time.time()
                        st.info(f"总耗时: {end_time - start_time:.2f} 秒")

    else:
        st.error("加载模型或设置 ChromaDB collection 失败。请检查日志和配置。")
else:
    st.error("初始化 ChromaDB 客户端失败。请检查日志。")


# --- 页脚/信息侧边栏 ---
st.sidebar.header("系统配置")
st.sidebar.markdown(f"**向量存储:** ChromaDB")
st.sidebar.markdown(f"**数据路径:** `{CHROMA_PERSIST_DIR}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")
st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**检索 Top K:** `{TOP_K}`")