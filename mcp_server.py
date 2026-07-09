import os
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 1. 強制將工作目錄切換到這個腳本所在的資料夾，確保能正確讀取 .env 與 data
os.chdir(Path(__file__).parent)

import rag_app 

mcp = FastMCP("Org_RAG_Server")

# 2. 將 print 加上 file=sys.stderr，這樣提示訊息就不會干擾 MCP 的 JSON 通訊
print("正在載入語料庫與模型，請稍候...", file=sys.stderr)
embedder, corpora, client = rag_app.build_corpora_and_clients()
print("載入完成！", file=sys.stderr)

org_mapping = {
    "msm": "勵友協會（就業輔導）",
    "tyad": "桃園輔具中心（輔具/補助）"
}

@mcp.tool()
def query_rag_database(org_key: str, question: str) -> str:
    """
    從多組織 RAG 系統中檢索資料並回答問題。
    參數：
    org_key: 組織代碼。只能輸入 "msm" (代表勵友協會) 或 "tyad" (代表桃園輔具中心)。
    question: 欲查詢的具體問題。
    """
    if org_key not in org_mapping:
        return f"檢索失敗：找不到組織代碼 {org_key}，請確認是否輸入 msm 或 tyad。"

    org_label = org_mapping[org_key]

    try:
        lang_code = "zh-Hant"
        
        if rag_app.is_small_talk(client, question):
            ans = rag_app.generate_small_talk(client, org_key, question, lang_code)
            return ans

        hits = rag_app.hybrid_search(org_key, question, rag_app.FIXED_TOP_K, embedder, corpora, client, relax=0, lang_code=lang_code)
        if not hits:
            for relax in (1, 2, 3):
                hits = rag_app.hybrid_search(org_key, question, rag_app.FIXED_TOP_K, embedder, corpora, client, relax=relax, lang_code=lang_code)
                if hits: break

        if hits:
            retrieved = [h[1] for h in hits]
            system_prompt = rag_app.ORG_REGISTRY[org_key]["system_prompt"]
            ans = rag_app.generate_answer(client, system_prompt, question, retrieved, memory_summary="", lang_code=lang_code)
            refs = "\n".join([f"[{i+1}] {h[0]}" for i, h in enumerate(hits)])
            return f"{ans}\n\n參考來源：\n{refs}"
        else:
            return "不好意思，目前沒有找到相關資料。"

    except Exception as e:
        return f"系統發生錯誤：{str(e)}"

if __name__ == "__main__":
    mcp.run()