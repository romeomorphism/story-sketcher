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
    language: str = "en"
    movie_language: str = "en"

class ChatRequest(BaseModel):
    user_message: str
    story_context: str = ""
    history: list = []
    language: str = "en"

class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en-US"  # Default to English

# --- 辅助函数：处理 Base64 图片 ---
def decode_image(base64_string):
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def get_language_config(language_code: str):
    normalized = (language_code or "en").lower()
    language_map = {
        "en": "English",
        "zh": "Chinese",
        "de": "German",
        "fr": "French"
    }
    selected = language_map.get(normalized, "English")
    return normalized, selected

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
        _, target_language = get_language_config(request.language)

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
            conversation_history = "\nCONVERSATION CONTEXT (Child & AI discussion - KEY TO UNDERSTANDING THE DRAWING):\n"
            conversation_history += "=" * 60 + "\n"
            # 只保留最近的对话以获得更强的上下文
            recent_msgs = request.history[-12:] if len(request.history) > 12 else request.history
            for msg in recent_msgs:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        # Clean up any HTML from content
                        clean_content = content.replace('<br>', ' ').replace('<br/>', ' ')
                        conversation_history += f"{role.upper()}: {clean_content}\n"
                        # 提取可能的关键对象或主题
                        if role.lower() == "user":
                            recent_topics.append(clean_content)
            conversation_history += "=" * 60 + "\n"
            # 添加对象提示
            if recent_topics:
                objects_mentioned = " | ".join(recent_topics[-3:])  # Last 3 user messages likely contain object hints
                conversation_history += f"KEY TOPICS LIKELY IN DRAWING: {objects_mentioned}\n"
            conversation_history += "\n"
        
        # 3. 构造 Prompt (提示词) - 根据所选语言输出故事和问题
        prompt_text = f"""
        You are an AI co-creation partner named "StorySketcher" for children aged 4-10.
        
        Current Story Context (The story so far):
        "{request.story_context}"
        
        {conversation_history}
        
        CRITICAL INSTRUCTION FOR OBJECT RECOGNITION:
        The child drew this picture based on the conversation history above. Your job is to:
        1. FIRST, carefully read and understand the conversation history
        2. IDENTIFY KEY OBJECTS/CHARACTERS mentioned in the conversation - these are likely what the child drew
        3. MATCH the visual elements in the drawing to the conversation context
        4. If the drawing shows shapes/colors, use conversation context to interpret what they represent
        5. When ambiguous, ALWAYS prefer the interpretation that matches the conversation history
        6. Reference specific things discussed in the conversation when explaining the drawing
        
        Example: If the conversation was about a "red dragon", and the drawing shows red shapes, interpret those shapes as the dragon mentioned in conversation.
        
        Task:
        1. (Vision Agent) Look at this drawing carefully. Identify the visual elements (shapes, colors, composition).
           Based on the conversation history, interpret what objects/characters the child intended to draw.
           Explain your interpretation by referencing the conversation.
        
        2. (Story Agent) Continue the story:
           - Build on the current story context
           - Incorporate the objects/characters the child drew (as identified through conversation context)
           - Include specific details from the conversation history
           - Use descriptive language to make the story engaging and vivid
              - Make the new story part 2-3 short sentences in {target_language} (suitable for 4-10 year olds)
           - Ensure smooth narrative flow from the current story
        
          3. (Socratic Agent) Ask an engaging question in {target_language} to guide the child's next creative step.
           The question should encourage them to draw what happens next in the story.
        
        Return strictly in JSON format:
        {{
            "story_update": "New story continuation (2-3 sentences)...",
            "ai_question": "Engaging question to guide next drawing..."
        }}
        """

        # 4. 调用 Doubao (使用 Ark 客户端)
        response = ark_client.chat.completions.create(
            model="doubao-seed-2-0-lite-260215",
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
        _, target_language = get_language_config(request.language)

        # 构建消息列表，包含对话历史
        messages = []
        
        # 添加系统消息
        system_message = f"""You are StoryBuddy, a friendly and enthusiastic AI storytelling assistant.
The user is a young child (4-10 years old).
    Reply in short, simple, and encouraging {target_language}. Keep the tone magical and fun."""
        
        messages.append({"role": "system", "content": system_message})
        
        # 添加之前的对话历史（将前端的 "ai" role 映射为 API 标准的 "assistant"）
        if request.history and len(request.history) > 0:
            for msg in request.history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "ai":
                        role = "assistant"
                    if role in ("user", "assistant") and content:
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
        print(f"Chat error: {type(e).__name__}: {e}")
        return {"reply": "Sorry, my magic brain fuzzy for a second. Can you say that again?"}

@app.post("/api/generate-movie")
async def generate_movie(request: DrawingRequest):
    """
    Animation Agent with Story Narration:
    1. 基于画作和故事生成动画
    2. 添加所选语言讲述故事的语音
    3. 创建完整的故事视频
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

        # Language selection: movie language follows app language by default
        fallback_lang = request.language if hasattr(request, "language") else "en"
        selected_language = (request.movie_language or fallback_lang or "en").lower()
        _, narration_language = get_language_config(selected_language)
            
        # Create enhanced prompt that includes story narration requirement
        # Ensure the video includes narration in selected language
        base_prompt = f"""Create a vivid, child-friendly animation based on this drawing:
        
STORY NARRATION (IMPORTANT):
    Add clear {narration_language} narration/voiceover that tells the following story:
"{story_text}"

    LANGUAGE RULES:
    - The narration language MUST be {narration_language}
    - If the story text is in another language, translate it into {narration_language} for narration
    - Keep wording simple and child-friendly in {narration_language}

VIDEO INSTRUCTIONS:
- Make the drawing items move lively and expressively
- Synchronize the animation with the story narration
- Keep the narration clear, slow, and simple for children aged 4-10
- The narration should match the animation and bring the story to life
- Use a warm, friendly, enthusiastic tone for the narration
- Duration: 10 seconds"""
        
        prompt_text = base_prompt if story_text else f"""Create a 10-second animation based on this drawing.
Make it lively and fun, suitable for children. 
Focus on the main characters and add simple movements like waving, jumping, or smiling.
    Add cheerful background sounds or music.
    Narration/voice language MUST be {narration_language}."""
        
        # Ensure prompt is not too long (API limits)
        if len(prompt_text) > 800:
            prompt_text = prompt_text[:800]
        
        print(f"Generating movie in language [{selected_language}] with narration prompt: {prompt_text}")

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
            generate_audio=True,  # Generate audio with story narration
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


@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Text-to-Speech API:
    Convert text to speech audio and return as base64 encoded audio data.
    This allows the chatbot buttons to play audio responses.
    """
    try:
        # Import requests for TTS API call
        import requests
        
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            return {"error": "Text cannot be empty"}
        
        # Truncate text if too long (TTS has character limits, typically 1000-2000 chars)
        text_to_convert = request.text[:500]  # Limit to 500 chars for efficiency
        
        # Use Ark client for TTS (if available) or use a simple TTS approach
        # For now, we'll use a text-to-speech approach via the Ark platform
        # or external TTS service
        
        try:
            # Try using Ark's TTS capability if available
            # This is a fallback - you may need to use a dedicated TTS API
            tts_response = ark_client.audio.speech.create(
                model="tts-model",  # Specify TTS model
                input=text_to_convert,
                voice="en-US-Neural2-A",  # English voice
                response_format="mp3"
            )
            
            # If successful, return the audio data
            return {
                "success": True,
                "audio": base64.b64encode(tts_response.content).decode('utf-8') if hasattr(tts_response, 'content') else tts_response
            }
        except:
            # Fallback: Use external TTS service (like Google Cloud TTS or similar)
            # For this example, we'll return a placeholder message
            print("Ark TTS not available, attempting alternative TTS method...")
            
            # You can integrate with:
            # 1. Google Cloud Text-to-Speech API
            # 2. Azure Text-to-Speech API
            # 3. Amazon Polly
            # 4. Other TTS services
            
            # For now, return a simple message asking to configure TTS
            return {
                "success": False,
                "message": "TTS service temporarily unavailable. Please configure a TTS API.",
                "text": text_to_convert
            }
    
    except Exception as e:
        print(f"TTS Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to convert text to speech"
        }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 