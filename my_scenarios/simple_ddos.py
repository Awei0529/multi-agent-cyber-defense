#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

from pettingzoo.utils import wrappers
from pettingzoo.mpe._mpe_utils.core import World, Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
import functools
import numpy as np

class Scenario(BaseScenario):
    def make_world(self, N=2, num_adversaries=3, np_random=np.random):
        world = World()
        world.dim_c = 2
        
        # 添加智能體
        world.agents = []
        # 添加攻擊者
        for i in range(num_adversaries):
            agent = Agent()
            agent.name = f"adversary_{i}"
            agent.adversary = True
            agent.collide = True
            agent.silent = True
            agent.size = 0.075
            agent.accel = 3.0
            agent.max_speed = 1.0
            world.agents.append(agent)
        # 添加防禦者
        for i in range(N):
            agent = Agent()
            agent.name = f"defender_{i}"
            agent.adversary = False
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 4.0
            agent.max_speed = 1.3
            world.agents.append(agent)
            
        # 添加landmarks (包括目標伺服器)
        world.landmarks = []
        # 添加中心伺服器
        landmark = Landmark()
        landmark.name = "central_server"
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.1
        landmark.color = np.array([0.1, 0.9, 0.1])  # 綠色
        world.landmarks.append(landmark)
        
        # 設置初始狀態
        self.reset_world(world, np_random)
        return world

    def reset_world(self, world, np_random):
        # 隨機設置每個智能體和目標
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])  # 紅色 (攻擊者)
                agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            else:
                agent.color = np.array([0.35, 0.35, 0.85])  # 藍色 (防禦者)
                agent.state.p_pos = np_random.uniform(-0.5, +0.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        # 中心伺服器位於原點
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.zeros(world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # 攻擊者獎勵: 接近中心伺服器
        # 防禦者獎勵: 阻止攻擊者接近中心伺服器
        reward = 0
        central_server = world.landmarks[0]
        
        if agent.adversary:
            # 攻擊者的獎勵是它接近中心伺服器
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - central_server.state.p_pos)))
            reward -= dist
        else:
            # 防禦者的獎勵是它阻止攻擊者接近中心伺服器
            for adv in [a for a in world.agents if a.adversary]:
                dist = np.sqrt(np.sum(np.square(adv.state.p_pos - central_server.state.p_pos)))
                reward += dist
                # 額外獎勵: 如果防禦者接近攻擊者
                def_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                if def_dist < 0.25:  # 如果防禦者足夠接近攻擊者
                    reward += 0.5
        
        return reward

    def observation(self, agent, world):
        # 獲取位置
        central_server = world.landmarks[0]
        
        # 目標伺服器位置
        entity_pos = [central_server.state.p_pos]
        
        # 其他智能體位置
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos)
            other_vel.append(other.state.p_vel)
            
        # 返回為: 自身位置, 自身速度, 目標位置, 其他智能體位置和速度
        return np.concatenate([agent.state.p_pos, agent.state.p_vel] + entity_pos + other_pos + other_vel)

def env(num_attackers=3, num_defenders=2, max_cycles=500, render_mode=None):
    scenario = Scenario()
    world = scenario.make_world(N=num_defenders, num_adversaries=num_attackers)
    return SimpleEnv(scenario, world, render_mode=render_mode, max_cycles=max_cycles)

def DDoSenv(num_attackers=3, num_defenders=2, max_cycles=500, render_mode="human"):
    """
    創建一個自定義的DDoS攻擊模擬環境
    
    參數:
    - num_attackers: 攻擊者數量
    - num_defenders: 防禦者數量
    - max_cycles: 最大步數
    - render_mode: 渲染模式
    
    返回:
    - env: PettingZoo環境
    """
    _env = functools.partial(
        env,
        num_attackers=num_attackers,
        num_defenders=num_defenders,
        max_cycles=max_cycles,
        render_mode=render_mode
    )
    
    environment = _env()
    environment = wrappers.OrderEnforcingWrapper(environment)
    
    return environment


# 測試函數
if __name__ == "__main__":
    env = DDoSenv()
    env.reset()
    print("Agents:", env.agents)
    
    # 簡單測試幾個步驟
    for step in range(5):
        print(f"Step {step+1}")
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()
            env.step(action)
            
    env.close()


# In[ ]:




