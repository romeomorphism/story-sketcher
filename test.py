'''
Author: Xinyu Liu
Date: 2026-02-11 00:02:20
LastEditors: Xinyu Liu
LastEditTime: 2026-02-11 00:02:42
FilePath: /StorySkecher/test.py
Description: 

Copyright (c) 2026 by Xinyu Liu, All Rights Reserved. 
'''
import requests
import os

# 替换为你的代理端口
PROXY_PORT = 8118
proxies = {
    "http": f"http://127.0.0.1:{PROXY_PORT}",
    "https": f"http://127.0.0.1:{PROXY_PORT}",
}

print(f"正在通过端口 {PROXY_PORT} 检查 IP 位置...")

try:
    # 强制通过代理访问 IP 查询网站
    response = requests.get("http://ip-api.com/json", proxies=proxies, timeout=5)
    data = response.json()
    
    print("\n----------------IP 检测结果----------------")
    print(f"你的真实 IP: {data.get('query')}")
    print(f"该 IP 所在国家: {data.get('country')}")
    print("-------------------------------------------")

    if data.get('country') in ['China', 'Hong Kong']:
        print("❌ 失败：你的代理没生效！Google 看到你在中国/香港。")
        print("💡 解决：请检查 VPN 软件是否开启，端口是否真的是 7890。")
    else:
        print("✅ 成功：现在你的 Python 看起来是在海外。")
        print("如果这时连 Google 还报错，请使用方案一 (transport='rest')。")

except Exception as e:
    print(f"❌ 连不上代理：{e}")
    print("请确认 VPN 软件已打开！")