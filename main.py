import os
import base64
import io
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
from volcenginesdkarkruntime import Ark 

# --- 配置部分 ---

# 1. API Keys (Prioritize environment variables for Vercel/Production)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ARK_API_KEY = os.environ.get("ARK_API_KEY")

# Is running on Vercel?
IS_VERCEL = os.environ.get("VERCEL") == "1"

# 2. Proxy Settings (Only for local development if needed)
if not IS_VERCEL:
    # Check if user has set a local proxy variable manually or rely on system
    # If you are running locally without VPN software, you might want to comment this out
    proxy_port = os.environ.get("PROXY_PORT", "8118") # Default to 8118 if not set
    # Only set if not already set by system
    if "HTTP_PROXY" not in os.environ:
         os.environ["HTTP_PROXY"] = f"http://127.0.0.1:{proxy_port}"
         os.environ["HTTPS_PROXY"] = f"http://127.0.0.1:{proxy_port}"

# 3. Configure Gemini (not needed for Doubao)
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not required for Doubao")

app = FastAPI(title="StorySketcher Gemini Backend")

# 5. 配置 Ark (动画生成引擎)
ark_client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3",
                 api_key=ARK_API_KEY)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 数据模型 ---
class DrawingRequest(BaseModel):
    image_base64: str
    story_context: str = ""
    history: list = []

class ChatRequest(BaseModel):
    user_message: str
    story_context: str = ""
    history: list = []

# --- 辅助函数：处理 Base64 图片 ---
def decode_image(base64_string):
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

# --- 核心接口 ---

@app.post("/api/analyze-drawing")
async def analyze_drawing(request: DrawingRequest):
    """
    Vision Agent + Story Agent (Using Doubao Model):
    1. 识别画作内容 (Vision)
    2. 生成故事片段 (Story)
    3. 提出苏格拉底式问题 (Socratic Scaffolding)
    
    现在支持对话历史，确保生成的故事与之前的对话和故事内容保持一致。
    重点：优先参考对话历史来理解画作中的对象，因为用户是根据对话内容来画的。
    """
    try:
        # 1. 解码图片并转换为 base64
        img = decode_image(request.image_base64)
        
        # 将图片保存为 PNG 格式并转为 base64
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # 2. 构建对话历史部分并提取关键信息
        conversation_history = ""
        recent_topics = []
        if request.history and len(request.history) > 0:
            conversation_history = "CONVERSATION HISTORY (Very Important - Use this to understand the objects in the drawing):\n"
            # 只保留最近的对话以获得更强的上下文
            recent_msgs = request.history[-10:] if len(request.history) > 10 else request.history
            for msg in recent_msgs:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        conversation_history += f"{role.upper()}: {content}\n"
                        # 提取可能的关键对象或主题
                        if role.lower() == "user":
                            recent_topics.append(content)
            conversation_history += "\n"
        
        # 3. 构造 Prompt (提示词) - 保持英文，强调对话历史的重要性
        prompt_text = f"""
        You are an AI co-creation partner named "StorySketcher" for children aged 4-10.
        
        Current Story Context (The story so far):
        "{request.story_context}"
        
        {conversation_history}
        
        CRITICAL INSTRUCTION:
        The user drew this picture based on the conversation history above. Therefore:
        1. FIRST, carefully read and understand the conversation history
        2. THEN, look at the drawing and MATCH the objects/characters in the drawing to what was discussed in the conversation
        3. Use the conversation context to correctly identify and interpret objects, characters, and themes in the drawing
        4. If something in the drawing is ambiguous, prefer the interpretation that matches the conversation history
        
        Task:
        1. (Vision Agent) Look at this drawing. Identify the main objects, colors, and setting. INTERPRET THEM IN THE CONTEXT OF THE PREVIOUS CONVERSATION.
        2. (Story Agent) Continue the story based on the new drawing and the conversation context. 
           - Ensure the new sentences flow naturally from the "Current Story Context".
           - Remember and reference key details from the conversation history
           - Make the story cohesive, logical, and consistent with what was discussed in the conversation
           - Add 1-2 short, simple sentences in English.
        3. (Socratic Agent) Ask a heuristic question in English to guide the child to draw what happens NEXT.
        
        Strictly return JSON format:
        {{
            "story_update": "Story continuation text in English...",
            "ai_question": "Question text in English..."
        }}
        """

        # 4. 调用 Doubao (使用 Ark 客户端)
        response = ark_client.chat.completions.create(
            model="doubao-seed-1-6-251015",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        # 5. 解析结果
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        return result

    except Exception as e:
        print(f"Error: {e}")
        # 错误信息改为英文
        return {
            "story_update": "Wow, what a great drawing! But I got a little distracted...",
            "ai_question": "Can you try again? Or tell me what you drew?"
        }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Story Agent (纯对话模式 - Using Doubao Model):
    继续与孩子对话，引导故事发展。
    支持对话历史，记住之前的对话内容。
    """
    try:
        # 构建消息列表，包含对话历史
        messages = []
        
        # 添加系统消息
        system_message = """You are StoryBuddy, a friendly and enthusiastic AI storytelling assistant.
The user is a young child (4-10 years old).
Reply in short, simple, and encouraging English. Keep the tone magical and fun."""
        
        messages.append({"role": "system", "content": system_message})
        
        # 添加之前的对话历史
        if request.history and len(request.history) > 0:
            for msg in request.history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        messages.append({"role": role, "content": content})
        
        # 添加当前用户消息和故事上下文
        story_context_str = request.story_context if request.story_context and len(request.story_context.strip()) > 0 else "None yet."
        
        current_user_message = f"""Current Story Content: "{story_context_str}"

User's input: "{request.user_message}"

Instructions:
1. Your reply MUST relate to the "Current Story Content" and remember what we discussed before.
2. If the user suggests a new idea, encourage them to draw it on the canvas to add it to the story.
3. Be consistent with anything you said before in the conversation."""
        
        messages.append({"role": "user", "content": current_user_message})
        
        # 调用 Doubao (使用 Ark 客户端)
        response = ark_client.chat.completions.create(
            model="doubao-seed-1-6-251015",
            messages=messages
        )
        
        return {"reply": response.choices[0].message.content}

    except Exception as e:
        print(f"Error: {e}")
        # 错误信息改为英文
        return {"reply": "Sorry, my magic brain fuzzy for a second. Can you say that again?"}

@app.post("/api/generate-movie")
async def generate_movie(request: DrawingRequest):
    """
    Animation Agent (占位符)
    """
    try:
        # Decode the drawn image
        img = decode_image(request.image_base64)
        
        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create movie from the drawing
        # Ensure we have the full data URI
        image_url = request.image_base64
        if not image_url.startswith("data:"):
            image_url = f"data:image/png;base64,{request.image_base64}"

        # Clean the story context if it contains placeholder text
        story_text = request.story_context.strip() if request.story_context else ""
        if "Once upon a time..." in story_text and len(story_text) < 30:
            # If it's just the placeholder, ignore it for prompt purposes
            story_text = ""
            
        # Use story context as prompt if available, otherwise fallback
        base_prompt = "Create a vivid, child-friendly animation based on this drawing. Make the drawing items move lively. The background narative is" + f" '{story_text}'."
        prompt_text = f"{base_prompt} Story context: {story_text}" if story_text else "Create a 10-second animation based on this drawing. Make it lively and fun, suitable for children. Focus on the main characters and add simple movements like waving, jumping, or smiling."
        
        # Ensure prompt is not too long (API limits)
        if len(prompt_text) > 500:
            prompt_text = prompt_text[:500]
        
        print(f"Generating movie with prompt: {prompt_text}")

        create_movie_response = ark_client.content_generation.tasks.create(
            model="doubao-seedance-1-5-pro-251215",
            content=[
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ],
            generate_audio=True,
            ratio="adaptive",
            duration=10,
            watermark=False,
        )
        
        return {"movie_task_id": create_movie_response.id}
    
    except Exception as e:
        print(f"Error: {e}")
        return {"error": "Failed to generate animation. Please try again."}

@app.get("/api/check-movie-status/{task_id}")
async def check_movie_status(task_id: str):
    try:
        print(f"----- polling task status for {task_id} -----")
        # Get task status
        response = ark_client.content_generation.tasks.get(task_id=task_id)
        
        # Check status (normalize to uppercase just in case)
        status = response.status
        if hasattr(status, 'upper'):
            status = status.upper()
        
        print(f"Current status: {status}")

        if status == "SUCCEEDED":
            print("----- task succeeded -----")
            # Log the full response content to debug structural issues
            import pprint
            try:
                # Attempt to convert to dict if possible, or just print object
                if hasattr(response, 'to_dict'):
                    pprint.pprint(response.to_dict())
                else:
                    print(response)
            except:
                print(response)

            # Create a simple representation for logging
            print(f"Task ID: {response.id}, Status: {status}")
            
            # Extract video content
            video_url = ""
            try:
                # Based on the debug output:
                # content is an object with a video_url attribute
                if hasattr(response, 'content'):
                    content = response.content
                    # It seems content might be an object that has video_url attribute directly
                    if hasattr(content, 'video_url'):
                        video_url = content.video_url
                    # Or it might be a dictionary if we are lucky (though the error says not subscriptable)
                    elif isinstance(content, dict):
                        video_url = content.get('video_url', "")
                    # Fallback for the complex structure we saw earlier, just in case
                    elif isinstance(content, list) and len(content) > 0:
                        video_url = content[0].video_source.url
            except Exception as extract_err:
                 print(f"Error extracting video URL: {extract_err}")
            
            print(f"Video URL: {video_url}")
            return {"status": "SUCCEEDED", "video_url": video_url}
            
        elif status == "FAILED":
            print("----- task failed -----")
            error_msg = getattr(response, 'error', 'Unknown error')
            print(f"Error: {error_msg}")
            return {"status": "FAILED"}
            
        else:
            # QUEUED, RUNNING
            print(f"Current status: {status}")
            return {"status": status} 

    except Exception as e:
        print(f"Error checking status: {e}")
        return {"status": "ERROR", "message": str(e)}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 