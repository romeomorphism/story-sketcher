'''
Author: Xinyu Liu
Date: 2026-02-10 23:47:17
LastEditors: Xinyu Liu
LastEditTime: 2026-02-11 00:03:48
FilePath: /StorySkecher/check_models.py
Description: 

Copyright (c) 2026 by Xinyu Liu, All Rights Reserved. 
'''
import os
import google.generativeai as genai
from google.generativeai.types import generation_types

# ================= 配置区域 =================

# 1. 填入你的 Google API Key
API_KEY = "AIzaSyBroeXQlN4KCEM4u8xUVi2_ycTfnp9AQts"

# 2. 【关键】设置代理端口
# 常见的代理软件默认端口：
# - Clash / ClashX: 7890
# - V2RayN / V2RayNG: 10808 或 1080
# - Shadowrocket / Surge: 6152 或 8888
# 如果你不确定，请打开你的 VPN 软件设置查看 "HTTP Proxy Port"
PROXY_PORT = 8118

# ===========================================

# 设置环境变量，强制流量走代理
proxy_url = f"http://127.0.0.1:{PROXY_PORT}"
os.environ["HTTP_PROXY"] = proxy_url
os.environ["HTTPS_PROXY"] = proxy_url

print(f"🔄 正在通过代理 {proxy_url} 连接 Google...")

try:
    # 配置 API
    genai.configure(api_key=API_KEY, transport="rest")

    print("📡 正在获取模型列表...")
    
    # 获取所有模型
    models = genai.list_models()
    
    found = False
    print("\n✅ 你的 API Key 可用的模型有：")
    print("-" * 30)
    for m in models:
        # 只显示支持内容生成的模型
        if 'generateContent' in m.supported_generation_methods:
            print(f"🌟 {m.name}")
            found = True
            
    if not found:
        print("⚠️ 连接成功，但没有找到支持 generateContent 的模型。")
        
    print("-" * 30)
    print("测试结束。如果看到上面的模型列表，说明你的代理设置成功了！")

except Exception as e:
    print("\n❌ 连接失败！")
    print(f"错误信息: {e}")
    print("\n💡 排查建议：")
    print(f"1. 请确认你的 VPN 软件已开启，且 HTTP 代理端口确实是 {PROXY_PORT}。")
    print("2. 请确认你的终端 (Terminal) 可以通过该端口访问网络。")
    print("3. 如果报错 'User location is not supported'，说明你的 VPN 节点还在香港或国内，请切换到 美国/日本/台湾 节点。")