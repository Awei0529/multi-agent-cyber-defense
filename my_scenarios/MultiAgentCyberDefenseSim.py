#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import requests
import numpy as np
from typing import Optional, List
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 設定模組路徑與導入自定義環境
sys.path.append("C:/Users/User/multiagent-cybersecurity")
from my_scenarios.simple_ddos import DDoSenv #模擬DDoS環境
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微軟雅黑字體
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號


#紅點:攻擊者
#綠點:中心目標(防守目標)
#藍點:防守者

# === 定義分層防禦架構 ===
class DefenseLayer:
    def __init__(self, name, agents=None):
        self.name = name
        self.agents = agents or []
        self.shared_information = {}
        
    def add_agent(self, agent_id):
        if agent_id not in self.agents:
            self.agents.append(agent_id)
            
    def share_information(self, key, value):
        self.shared_information[key] = value
        
    def get_information(self, key):
        return self.shared_information.get(key)

# === 智能體間的資訊共享機制 ===
class AgentCommunicationProtocol:
    def __init__(self):
        self.messages = {}
        self.threat_information = {}
        
    def broadcast_message(self, sender, message_type, content):
        """廣播訊息給所有智能體"""
        if message_type not in self.messages:
            self.messages[message_type] = []
        self.messages[message_type].append({
            "sender": sender,
            "content": content,
            "timestamp": time.time()
        })
        
    def report_threat(self, agent_id, attacker_id, threat_level, position):
        """回報威脅資訊"""
        self.threat_information[attacker_id] = {
            "reporter": agent_id,
            "threat_level": threat_level,
            "position": position,
            "timestamp": time.time()
        }
        
    def get_highest_threats(self, count=3):
        """獲取最高威脅等級的攻擊者"""
        sorted_threats = sorted(
            self.threat_information.items(),
            key=lambda x: self._threat_level_to_value(x[1]["threat_level"]),
            reverse=True
        )
        return sorted_threats[:count]
    
    def _threat_level_to_value(self, level):
        """將威脅等級轉換為數值"""
        levels = {"極高": 4, "高": 3, "中": 2, "低": 1}
        return levels.get(level, 0)

# === 自定義 NVIDIA API 的 LangChain LLM 接口 ===
class NvidiaLLM(LLM):
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    model: str = "meta/llama-4-maverick-17b-128e-instruct" #可在此替換測試模型
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 1.0,
            "stream": False
        }
        response = requests.post(self.base_url, headers=headers, json=payload)
        try:
            result = response.json()
            output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return enforce_stop_tokens(output, stop) if stop else output
        except requests.exceptions.JSONDecodeError:
            print("\n❌ JSON 解碼失敗！以下是伺服器回應內容：")
            print("狀態碼:", response.status_code)
            print("回應文字:", response.text)
            return "⚠️ 模型回應解析失敗，請檢查 API 設定與輸入格式。"
    
    @property
    def _llm_type(self) -> str:
        return "nvidia-chat-completions"

# === 環境可視化函數 ===
def visualize_environment(environment, agent_positions):
    # 創建圖形
    plt.figure(figsize=(10, 10))
    
    # 繪製中心目標
    central_server = environment.unwrapped.world.landmarks[0].state.p_pos
    plt.scatter(central_server[0], central_server[1], color='green', s=200, marker='o', label='中心目標')
    
    # 繪製防禦者
    defenders_x = []
    defenders_y = []
    for agent, data in agent_positions.items():
        if "defender" in agent:
            x, y = data["position"]
            defenders_x.append(x)
            defenders_y.append(y)
    plt.scatter(defenders_x, defenders_y, color='blue', s=100, marker='s', label='防禦者')
    
    # 繪製攻擊者
    attackers_x = []
    attackers_y = []
    for agent, data in agent_positions.items():
        if "adversary" in agent:
            x, y = data["position"]
            attackers_x.append(x)
            attackers_y.append(y)
    plt.scatter(attackers_x, attackers_y, color='red', s=100, marker='^', label='攻擊者')
    
    # 添加圓形防禦區域指示
    circle = plt.Circle((central_server[0], central_server[1]), 0.5, color='green', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    # 設置圖形屬性
    plt.grid(True)
    plt.title('DDoS Defense Simulation Environment', fontsize=16)
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
    plt.legend(loc='upper right')
    
    # 設置坐標軸範圍
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    # 添加威脅距離標註
    for agent, data in agent_positions.items():
        if "adversary" in agent:
            x, y = data["position"]
            dist = calculate_distance((x, y), central_server)
            plt.annotate(f"{dist:.2f}", (x, y), xytext=(5, 5), textcoords='offset points')
    
    # 顯示圖形
    plt.savefig('ddos_simulation.png')  # 保存圖片
    plt.show()

# === 分層防禦可視化函數 ===
def visualize_environment_with_layers(environment, agent_positions, layer_assignment):
    # 創建圖形
    plt.figure(figsize=(10, 10))
    
    # 繪製中心目標
    central_server = environment.unwrapped.world.landmarks[0].state.p_pos
    plt.scatter(central_server[0], central_server[1], color='green', s=200, marker='o', label='中心目標')
    
    # 繪製不同層的防禦者
    layer_colors = {
        "perception": "blue",
        "analysis": "cyan",
        "response": "purple"
    }
    
    for layer, color in layer_colors.items():
        x_coords = []
        y_coords = []
        for agent in layer_assignment[layer]:
            if agent in agent_positions:
                x, y = agent_positions[agent]["position"]
                x_coords.append(x)
                y_coords.append(y)
        plt.scatter(x_coords, y_coords, color=color, s=100, marker='s', label=f'{layer}層防禦者')
    
    # 繪製攻擊者
    attackers_x = []
    attackers_y = []
    for agent, data in agent_positions.items():
        if "adversary" in agent:
            x, y = data["position"]
            attackers_x.append(x)
            attackers_y.append(y)
    plt.scatter(attackers_x, attackers_y, color='red', s=100, marker='^', label='攻擊者')
    
    # 添加防禦層圓圈
    circles = [
        plt.Circle((central_server[0], central_server[1]), 0.3, color='purple', fill=False, linestyle='-', alpha=0.5),
        plt.Circle((central_server[0], central_server[1]), 0.6, color='cyan', fill=False, linestyle='--', alpha=0.5),
        plt.Circle((central_server[0], central_server[1]), 0.9, color='blue', fill=False, linestyle=':', alpha=0.5)
    ]
    
    for circle in circles:
        plt.gca().add_patch(circle)
    
    # 設置圖形屬性
    plt.grid(True)
    plt.title('分層防禦DDoS模擬環境', fontsize=16)
    plt.xlabel('X座標', fontsize=12)
    plt.ylabel('Y座標', fontsize=12)
    plt.legend(loc='upper right')
    
    # 設置坐標軸範圍
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    # 顯示圖形
    plt.savefig('layered_defense_simulation.png')
    plt.show()

# === 初始化 NVIDIA LLM ===
llm = NvidiaLLM(api_key="--------")  # << 請替換為你的 NVIDIA API key
parser = StrOutputParser()

# === 定義 Prompt Template（以 agents 狀態進行總結）===
summary_prompt = PromptTemplate.from_template("""
你是一位資安分析師，請依據以下從模擬環境收集的 agents 狀態與座標資料，進行態勢分析。資料格式為結構化清單，每位 agent 包含其名稱、角色（防禦者、攻擊者或中心目標），與其座標 (x, y)。

請根據這些資料具體分析網路安全情勢，例如是否有攻擊者接近中心節點、防禦者的部署是否有效等。

以下為觀察資料：
{agent_states}

請每一輪都要清楚說明，不可省略：
1. 哪些攻擊者可能構成威脅？請計算攻擊者與中心目標的距離並排序。
2. 防禦者是否成功圍繞或保護中心節點？
3. 整體的安全評估與可能的建議行動。
""")

# === 定義防禦策略生成的 Prompt Template ===
defense_strategy_prompt = PromptTemplate.from_template("""
你是一位網路安全防禦策略專家。請根據以下環境狀態，為每個防禦者生成最佳的防禦策略。

環境狀態：
{agent_states}

請為每個防禦者生成一個動作策略，格式如下：
defender_0: [動作代碼] - [簡短理由]
defender_1: [動作代碼] - [簡短理由]
...

動作代碼說明：
0: 向左移動
1: 向下移動
2: 向右移動
3: 向上移動
4: 不移動

策略目標：
1. 優先攔截最接近中心目標的攻擊者
2. 形成圍繞中心目標的防禦陣型
3. 確保防禦者之間的協作，避免聚集在同一位置

請根據每個防禦者的當前位置、附近攻擊者的威脅程度，以及與其他防禦者的相對位置，生成最佳的防禦策略。
""")

# === 定義分層防禦策略的 Prompt Template ===
layered_defense_prompt = PromptTemplate.from_template("""
你是一位網路安全防禦策略專家，專精於分層防禦架構。請根據以下環境狀態，為三層防禦架構中的每個防禦者生成最佳的防禦策略。

環境狀態：
{agent_states}

防禦層分配：
感知層(外層): {perception_agents}
分析層(中層): {analysis_agents}
回應層(內層): {response_agents}

請為每個防禦者生成一個動作策略，格式如下：
[防禦者ID]: [動作代碼] - [簡短理由]

動作代碼說明：
0: 向左移動
1: 向下移動
2: 向右移動
3: 向上移動
4: 不移動

各層職責：
- 感知層：在外圍巡邏，監測並報告威脅
- 分析層：評估威脅優先級，協調防禦資源
- 回應層：直接攔截高優先級威脅，保護中心目標

請根據每個防禦者所屬的防禦層、當前位置、附近攻擊者的威脅程度，以及與其他防禦者的相對位置，生成最佳的防禦策略。
""")

# === 定義 LangChain Runnable Pipeline ===
summary_chain = (
    RunnableMap({
        "agent_states": lambda x: x["agent_states"],
    }) |
    summary_prompt |
    llm |
    parser
)

defense_strategy_chain = (
    RunnableMap({
        "agent_states": lambda x: x["agent_states"],
    }) |
    defense_strategy_prompt |
    llm |
    parser
)

layered_defense_chain = (
    RunnableMap({
        "agent_states": lambda x: x["agent_states"],
        "perception_agents": lambda x: x["perception_agents"],
        "analysis_agents": lambda x: x["analysis_agents"],
        "response_agents": lambda x: x["response_agents"],
    }) |
    layered_defense_prompt |
    llm |
    parser
)

# === 啟動 PettingZoo 環境進行模擬 ===
environment = DDoSenv(num_attackers=10, num_defenders=10, max_cycles=50) #模擬DDoS環境
environment.reset()
print("Agents:", environment.agents)

# 定義角色映射 - 在simple_adversary環境中，通常第一個agent是adversary，其他是defenders
# 這將根據agent名稱確定正確的角色
def determine_role(agent_name):
    if "adversary" in agent_name:
        return "攻擊者"
    else:
        return "防禦者"

# 中心目標的位置 (在simple_adversary環境中通常是(0,0))
center_position = (0, 0)
center_node = "central_server"
print(f"\n🔒 防禦者開始保護中心節點: {center_node}")

# 初始化防禦層和通訊協議
defense_layers = {
    "perception": DefenseLayer("感知層"),
    "analysis": DefenseLayer("分析層"),
    "response": DefenseLayer("回應層")
}

# 初始化通訊協議
communication_protocol = AgentCommunicationProtocol()

# 分配防禦者到不同防禦層
def assign_defenders_to_layers(defenders_list):
    """將防禦者分配到不同防禦層"""
    total = len(defenders_list)
    perception_count = total // 3
    analysis_count = total // 3
    response_count = total - perception_count - analysis_count
    
    layer_assignment = {
        "perception": [],
        "analysis": [],
        "response": []
    }
    
    for i, defender in enumerate(defenders_list):
        if i < perception_count:
            defense_layers["perception"].add_agent(defender)
            layer_assignment["perception"].append(defender)
        elif i < perception_count + analysis_count:
            defense_layers["analysis"].add_agent(defender)
            layer_assignment["analysis"].append(defender)
        else:
            defense_layers["response"].add_agent(defender)
            layer_assignment["response"].append(defender)
            
    return layer_assignment

# 更新防禦者決策邏輯
def decide_defender_action(agent, obs, layer, attackers_positions):
    """根據防禦層角色決定防禦者行動"""
    agent_pos = obs[:2]
    
    if layer == "perception":
        # 感知層：負責監測並報告威脅
        # 在中心目標周圍形成外圍巡邏
        angle = np.random.uniform(0, 2*np.pi)
        target_pos = np.array([
            center_position[0] + 0.8 * np.cos(angle),
            center_position[1] + 0.8 * np.sin(angle)
        ])
        
        # 檢測附近的攻擊者並報告
        for attacker, data in attackers_positions.items():
            attacker_pos = data["position"]
            dist = calculate_distance(agent_pos, attacker_pos)
            if dist < 0.5:
                # 發現攻擊者，報告威脅
                dist_to_center = calculate_distance(attacker_pos, center_position)
                if dist_to_center < 0.3:
                    threat_level = "極高"
                elif dist_to_center < 0.6:
                    threat_level = "高"
                elif dist_to_center < 1.0:
                    threat_level = "中"
                else:
                    threat_level = "低"
                
                communication_protocol.report_threat(
                    agent, attacker, threat_level, attacker_pos
                )
                communication_protocol.broadcast_message(
                    agent, "threat_detected",
                    f"發現攻擊者 {attacker} 位於 {attacker_pos}，威脅等級: {threat_level}"
                )
                
    elif layer == "analysis":
        # 分析層：評估威脅並協調防禦
        # 在中心目標周圍形成中層防禦圈
        angle = np.random.uniform(0, 2*np.pi)
        target_pos = np.array([
            center_position[0] + 0.5 * np.cos(angle),
            center_position[1] + 0.5 * np.sin(angle)
        ])
        
        # 分析最高威脅並更新共享資訊
        highest_threats = communication_protocol.get_highest_threats(3)
        if highest_threats:
            defense_layers["analysis"].share_information(
                "priority_targets", highest_threats
            )
            communication_protocol.broadcast_message(
                agent, "threat_analysis",
                f"優先防禦目標已更新: {[t[0] for t in highest_threats]}"
            )
            
    else:  # response layer
        # 回應層：直接攔截威脅
        # 獲取分析層提供的優先目標
        priority_targets = defense_layers["analysis"].get_information("priority_targets")
        
        if priority_targets and len(priority_targets) > 0:
            # 選擇最近的高優先級目標進行攔截
            target_attacker = None
            min_distance = float('inf')
            
            for attacker_id, threat_info in priority_targets:
                if attacker_id in attackers_positions:
                    attacker_pos = attackers_positions[attacker_id]["position"]
                    dist = calculate_distance(agent_pos, attacker_pos)
                    if dist < min_distance:
                        min_distance = dist
                        target_attacker = attacker_id
            
            if target_attacker:
                target_pos = attackers_positions[target_attacker]["position"]
                communication_protocol.broadcast_message(
                    agent, "response_action",
                    f"攔截攻擊者 {target_attacker}"
                )
            else:
                # 如果沒有優先目標，保護中心
                target_pos = center_position
        else:
            # 沒有分析資訊時，保護中心
            target_pos = center_position
    
    # 根據目標位置決定移動方向
    direction = np.array(target_pos) - np.array(agent_pos)
    
    # 選擇移動方向
    if abs(direction[0]) > abs(direction[1]):  # x方向更重要
        action = 0 if direction[0] < 0 else 2  # 左或右
    else:  # y方向更重要
        action = 1 if direction[1] < 0 else 3  # 下或上
        
    return action

# 用於存儲每個agent的最新狀態
agent_positions = {}
defense_strategies = {}  # 存儲每個防禦者的策略

# 解析AI生成的防禦策略
def parse_defense_strategies(strategy_text):
    strategies = {}
    lines = strategy_text.strip().split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            defender_id = parts[0].strip()
            strategy_part = parts[1].strip()
            
            # 提取動作代碼
            try:
                action_code = int(strategy_part.split()[0])
                if 0 <= action_code <= 4:  # 確保動作代碼有效
                    strategies[defender_id] = action_code
            except (ValueError, IndexError):
                # 如果解析失敗，使用默認策略
                continue
    
    return strategies

# 計算兩點之間的距離
def calculate_distance(pos1, pos2):
    return np.sqrt(np.sum(np.square(np.array(pos1) - np.array(pos2))))

# 獲取最近的攻擊者
def get_nearest_attacker(defender_pos, attackers_positions):
    nearest_attacker = None
    min_distance = float('inf')
    
    for attacker, data in attackers_positions.items():
        distance = calculate_distance(defender_pos, data["position"])
        if distance < min_distance:
            min_distance = distance
            nearest_attacker = attacker
    
    return nearest_attacker, min_distance

print("\n===== Step 1 =====")

# 收集環境狀態
agent_states = ""

# 加入 landmark 資訊（包含中心目標與導航地標）
for i, landmark in enumerate(environment.unwrapped.world.landmarks):
    x, y = landmark.state.p_pos
    if i == 0:
        landmark_role = "中心目標"
        name = "central_server"
    else:
        landmark_role = "導航地標"
        name = f"landmark_{i}"

    agent_states += f"Agent 名稱：{name}\n角色：{landmark_role}\n座標位置：(x={x:.2f}, y={y:.2f})\n---\n"

# 更新所有agent的位置信息
attackers_positions = {}
defenders_positions = {}

for agent in environment.agent_iter():
    obs, reward, termination, truncation, info = environment.last()

    # 儲存每個agent的觀察數據
    if obs is not None and len(obs) >= 2:  # 確保obs包含座標信息
        position_data = {
            "position": (obs[0], obs[1]),
            "role": determine_role(agent)
        }
        agent_positions[agent] = position_data
        
        # 分類存儲攻擊者和防禦者位置
        if "adversary" in agent:
            attackers_positions[agent] = position_data
        else:
            defenders_positions[agent] = position_data

    if termination or truncation:
        environment.step(None)
        continue

    # 動作決策 - 這裡只收集數據，不實際執行動作
    environment.step(environment.action_space(agent).sample())

# 加入 agents 位置資訊到狀態描述
for agent, data in agent_positions.items():
    x, y = data["position"]
    role = data["role"]
    agent_states += f"Agent 名稱：{agent}\n角色：{role}\n座標位置：(x={x:.2f}, y={y:.2f})\n---\n"

# 分配防禦者到不同防禦層
defenders_list = [agent for agent in environment.agents if "defender" in agent]
layer_assignment = assign_defenders_to_layers(defenders_list)

print("\n🛡️ 防禦者分層配置:")
for layer, agents in layer_assignment.items():
    print(f"{layer}: {', '.join(agents)}")

# 生成分層防禦策略
print("\n🛡️ 生成分層防禦策略...")
layered_strategy_input = {
    "agent_states": agent_states,
    "perception_agents": ", ".join(layer_assignment["perception"]),
    "analysis_agents": ", ".join(layer_assignment["analysis"]),
    "response_agents": ", ".join(layer_assignment["response"])
}
layered_strategy_output = layered_defense_chain.invoke(layered_strategy_input)
print("\n🤖 分層防禦策略：\n", layered_strategy_output)

# 可視化分層防禦環境
print("\n🖼️ 生成分層防禦環境可視化...")
visualize_environment_with_layers(environment, agent_positions, layer_assignment)

# 威脅分析表格
print("\n📊 攻擊者威脅分析")
print("-" * 50)
print(f"{'攻擊者':^15}|{'距離中心目標':^15}|{'威脅等級':^15}")
print("-" * 50)

# 計算每個攻擊者的威脅等級
central_server = environment.unwrapped.world.landmarks[0].state.p_pos
threat_data = []

for agent, data in attackers_positions.items():
    pos = data["position"]
    dist = calculate_distance(pos, central_server)
    
    # 根據距離判斷威脅等級
    if dist < 0.3:
        threat_level = "極高"
    elif dist < 0.6:
        threat_level = "高"
    elif dist < 1.0:
        threat_level = "中"
    else:
        threat_level = "低"
        
    threat_data.append((agent, dist, threat_level))

# 按距離排序並輸出
for agent, dist, level in sorted(threat_data, key=lambda x: x[1]):
    print(f"{agent:^15}|{dist:^15.2f}|{level:^15}")

print("-" * 50)

# 分析當前環境狀態
print("\n📡 請求 NVIDIA LLM 模型分析 agent 狀態...")
llm_input = {"agent_states": agent_states}
output = summary_chain.invoke(llm_input)
print("\n🤖 模型回應：\n", output)

# 顯示分層防禦的通訊記錄
print("\n📨 防禦層通訊記錄:")
for message_type, messages in communication_protocol.messages.items():
    print(f"\n{message_type.upper()}:")
    for msg in messages:
        print(f"  {msg['sender']}: {msg['content']}")

environment.close()
print("\n✅ 模擬完成")


# In[ ]:




