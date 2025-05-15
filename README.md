# 🛡️ Multi-Agent Cyber Defense Simulation Platform

本專案結合自訂 DDoS 多智能體環境與 AI 驅動防禦策略，提供一個專為資安防禦、AI 協同決策與分層防禦架構設計的模擬平台。整合了多智能體模擬（PettingZoo）、大型語言模型（LLM）推理、分層防禦機制與智能體協作，可應用於學術研究與教學。

---

## 📂 專案結構

| 檔案名稱 | 說明 |
|----------|------|
| `simple_ddos.py` | 多智能體 DDoS 環境，模擬攻擊者、防禦者與中心目標的動態交互。 |
| `MultiAgentCyberDefenseSim.py` | 主模擬程式，整合分層防禦、AI 策略生成、LLM 決策、智能體協作與可視化功能。 |

---

## 🔍 主要特色

### 🧠 自訂多智能體 DDoS 環境
- 基於 PettingZoo 的 AEC 多智能體框架
- 支援彈性設定攻擊者、防禦者數量
- 支援多角色互動（攻擊者、防禦者、中心目標）
- 空間動態與策略性攻防模擬

### 🧱 分層防禦架構
- 分為感知層、分析層與回應層
- 實作智能體分工、通訊與協同
- 模擬真實世界的縱深防禦模型

### 🤖 AI/LLM 驅動策略生成
- 整合 NVIDIA LLM API（如 Llama 系列）進行：
  - 威脅分析
  - 防禦策略生成
  - 態勢總結
- 可測試不同 LLM（如 Llama 3.1 vs Llama 4）在資安任務上的表現

### 🛰️ 智能體協作與通訊
- 威脅情報共享
- 廣播式通訊協定
- 協同防禦與目標優先級策略

### 📊 可視化支援
- 繪製智能體位置、防禦分層、威脅等級等圖表
- 支援中文標籤，方便教學與展示

---

## 🧪 學術應用情境

- **AI 資安研究**：評估多智能體與 LLM 策略的防禦效果
- **分層防禦建模**：模擬縱深防禦與智能體協同運作
- **資安教育**：輔助教學的互動式攻防模擬平台
- **LLM 能力評測**：比較不同模型在策略推理上的效果

---

## ⚙️ 安裝與執行

### ✅ 環境需求
- Python 3.8+
- [PettingZoo](https://pettingzoo.farama.org/)
- [LangChain](https://www.langchain.com/)
- matplotlib, numpy, requests
- NVIDIA LLM API Key（建議使用 Llama 系列）

### 📦 安裝方式

```bash
pip install pettingzoo langchain matplotlib numpy requests
```

▶️ 執行方式
(A) 執行 DDoS 多智能體模擬
```bash
python simple_ddos.py
```
觀察多智能體隨機行為與基本攻防動態。

(B) 執行完整 AI 驅動防禦模擬
```bash
python MultiAgentCyberDefenseSim.py
```
執行單步分層防禦、LLM 分析與圖形化繪製。

📤 模擬輸出
智能體狀態與威脅等級表格

LLM 產生的防禦策略與情勢分析

圖形化輸出：

ddos_simulation.png

layered_defense_simulation.png

分層通訊與防禦紀錄（文字輸出）

🔧 客製化與擴充
可自訂攻擊者、防禦者數量與模擬步數

修改 simple_ddos.py 中的獎勵函數與行為策略

更換不同的 LLM 模型或 Prompt 進行對比實驗

🤝 貢獻方式
歡迎提出：

Issue

Pull Request

新功能建議或錯誤回報！

🙏 致謝
PettingZoo 多智能體模擬框架

LangChain LLM 管線整合

NVIDIA Llama API

📬 聯絡方式
作者資訊
作者：王承偉(WANG CHENG-WEI)

系所：資訊管理系（應用於研究所考試作品）

聯絡：可於 GitHub Issues 討論

本專案僅用於學術研究與技術展示，未用於任何實際投資建議。請遵守相關資料來源的使用條款。
