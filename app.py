import base64
import json
from datetime import datetime
import gradio as gr
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import ast
import os
from datetime import datetime
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
import logging  # 添加在文件顶部的import区域

# Define constants
DESCRIPTION = "[ShowUI Demo](https://huggingface.co/showlab/ShowUI-2B)"
_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1344 * 28 * 28

# Specify the model repository and destination folder
# model_repo = "showlab/ShowUI-2B"
destination_folder = "./showui-2b"
# modelscope download --model 'Qwen/Qwen2-7b' --local_dir 'path/to/dir'
# modelscope download --model AI-ModelScope/ShowUI-2B --local_dir 'L:\Gitee\ShowUI\showui-2b'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# List all files in the repository
# files = list_repo_files(repo_id=model_repo)

# Download each file to the destination folder
# for file in files:
#     file_path = hf_hub_download(repo_id=model_repo, filename=file, local_dir=destination_folder)
#     print(f"Downloaded {file} to {file_path}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "showui-2b",
    # "showlab/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
# modelscope download --model Qwen/Qwen2-VL-2B-Instruct
# modelscope download --model Qwen/Qwen2-VL-2B-Instruct --local_dir 'L:\Gitee\ShowUI\Qwen\Qwen2-VL-2B-Instruct'

# Helper functions
def draw_point(image_input, point=None, radius=5):
    """Draw a point on the image."""
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = Image.fromarray(np.uint8(image_input))

    if point:
        x, y = point[0] * image.width, point[1] * image.height
        ImageDraw.Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
    return image

def array_to_image_path(image_array):
    """Save the uploaded image and return its path."""
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    img = Image.fromarray(np.uint8(image_array))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    img.save(filename)
    return os.path.abspath(filename)

# @spaces.GPU
def run_showui(image, query):
    """主推理函数：根据输入的屏幕截图和文本指令，预测点击坐标"""
    # 新增日志模块
    import logging
    from datetime import datetime
    
    # 初始化日志格式
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    start_time = datetime.now()
    logging.info(f"▶▶ 开始推理流程 | 输入query: {query[:50]}...")

    try:
        # 图像处理阶段
        logging.info("正在保存上传图片...")
        image_path = array_to_image_path(image)
        logging.debug(f"临时图片路径: {image_path}")

        # 构建多模态输入信息（系统提示 + 图像元数据 + 用户query）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _SYSTEM},  # 系统级指令
                    # 图像元数据（路径 + 分辨率约束）
                    {"type": "image", "image": image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                    {"type": "text", "text": query}      # 用户输入的文本指令
                ],
            }
        ]
        
        logging.debug(f"多模态输入信息: {messages}")

        # 准备模型输入
        global model
        model = model.to("cuda")  # 将模型加载到GPU显存
        
        # 使用processor处理多模态输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 格式化对话模板
        logging.debug(f"格式化后的对话模板: {text}")
        image_inputs, video_inputs = process_vision_info(messages)  # 提取视觉特征
        logging.debug(f"视觉特征提取结果: {image_inputs}, {video_inputs}")
        logging.info("正在生成预测结果...")
        
        # 将输入数据转换为PyTorch张量
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,          # 自动填充到相同长度
            return_tensors="pt"    # 返回PyTorch张量
        ).to("cuda")               # 将输入数据移动到GPU

        # 生成预测结果（限制最大生成长度为128 tokens）
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        logging.debug(f"生成的token IDs: {generated_ids}")
        
        # 修剪生成结果：移除输入部分的token
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        logging.debug(f"修剪后的token IDs: {generated_ids_trimmed}")
        
        # 解码生成结果为可读文本
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,          # 跳过特殊token
            clean_up_tokenization_spaces=False # 保留原始空格
        )[0]
        logging.info(f"生成的文本: {output_text}")

        # 将输出文本解析为坐标列表（例如 "[0.35, 0.7]" → [0.35, 0.7]）
        click_xy = ast.literal_eval(output_text)
        logging.debug(f"解析后的坐标: {click_xy}")

        # 在原始图像上绘制红色标记点（半径10像素）
        result_image = draw_point(image_path, click_xy, radius=10)

        # 统计耗时
        total_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"◀◀ 完成推理 | 耗时: {total_time:.2f}s")
        
        return result_image, str(click_xy)

    except Exception as e:
        logging.error(f"推理失败: {str(e)}", exc_info=True)
        raise

# Function to record votes
def record_vote(vote_type, image_path, query, action_generated):
    """Record a vote in a JSON file."""
    vote_data = {
        "vote_type": vote_type,
        "image_path": image_path,
        "query": query,
        "action_generated": action_generated,
        "timestamp": datetime.now().isoformat()
    }
    with open("votes.json", "a") as f:
        f.write(json.dumps(vote_data) + "\n")
    return f"Your {vote_type} has been recorded. Thank you!"

# Helper function to handle vote recording
def handle_vote(vote_type, image_path, query, action_generated):
    """Handle vote recording by using the consistent image path."""
    if image_path is None:
        return "No image uploaded. Please upload an image before voting."
    return record_vote(vote_type, image_path, query, action_generated)

# Load logo and encode to Base64
with open("./assets/showui.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")


# Define layout and UI
def build_demo(embed_mode, concurrency_count=1):
    with gr.Blocks(title="ShowUI Demo", theme=gr.themes.Default()) as demo:
        # State to store the consistent image path
        state_image_path = gr.State(value=None)

        if not embed_mode:
            gr.HTML(
                f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <!-- Image -->
                    <div style="display: flex; justify-content: center;">
                        <img src="data:image/png;base64,{base64_image}" alt="ShowUI" width="320" style="margin-bottom: 10px;"/>
                    </div>
            
                    <!-- Description -->
                    <p>ShowUI is a lightweight vision-language-action model for GUI agents.</p>
            
                    <!-- Links -->
                    <div style="display: flex; justify-content: center; gap: 15px; font-size: 20px;">
                        <a href="https://huggingface.co/showlab/ShowUI-2B" target="_blank">
                            <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ShowUI--2B-blue" alt="model"/>
                        </a>
                        <a href="https://arxiv.org/abs/2411.17465" target="_blank">
                            <img src="https://img.shields.io/badge/arXiv%20paper-2411.17465-b31b1b.svg" alt="arXiv"/>
                        </a>
                        <a href="https://github.com/showlab/ShowUI" target="_blank">
                            <img src="https://img.shields.io/badge/GitHub-ShowUI-black" alt="GitHub"/>
                        </a>
                    </div>
                </div>
                """
            )

        with gr.Row():
            with gr.Column(scale=3):
                # Input components
                imagebox = gr.Image(type="numpy", label="Input Screenshot")
                textbox = gr.Textbox(
                    show_label=True,
                    placeholder="Enter a query (e.g., 'Click Nahant')",
                    label="Query",
                )
                submit_btn = gr.Button(value="Submit", variant="primary")

                # Placeholder examples
                gr.Examples(
                    examples=[
                        ["./examples/app_store.png", "Download Kindle."],
                        ["./examples/ios_setting.png", "Turn off Do not disturb."],
                        ["./examples/apple_music.png", "Star to favorite."],
                        ["./examples/map.png", "Boston."],
                        ["./examples/wallet.png", "Scan a QR code."],
                        ["./examples/word.png", "More shapes."],
                        ["./examples/web_shopping.png", "Proceed to checkout."],
                        ["./examples/web_forum.png", "Post my comment."],
                        ["./examples/safari_google.png", "Click on search bar."],
                    ],
                    inputs=[imagebox, textbox],
                    examples_per_page=3
                )

            with gr.Column(scale=8):
                # Output components
                output_img = gr.Image(type="pil", label="Output Image")
                # Add a note below the image to explain the red point
                gr.HTML(
                    """
                    <p><strong>Note:</strong> The <span style="color: red;">red point</span> on the output image represents the predicted clickable coordinates.</p>
                    """
                )
                output_coords = gr.Textbox(label="Clickable Coordinates")

                # Buttons for voting, flagging, regenerating, and clearing
                with gr.Row(elem_id="action-buttons", equal_height=True):
                    vote_btn = gr.Button(value="👍 Vote", variant="secondary")
                    downvote_btn = gr.Button(value="👎 Downvote", variant="secondary")
                    flag_btn = gr.Button(value="🚩 Flag", variant="secondary")
                    regenerate_btn = gr.Button(value="🔄 Regenerate", variant="secondary")
                    clear_btn = gr.Button(value="🗑️ Clear", interactive=True)  # Combined Clear button

            # Define button actions
            def on_submit(image, query):
                """Handle the submit button click."""
                if image is None:
                    raise ValueError("No image provided. Please upload an image before submitting.")
                
                # Generate consistent image path and store it in the state
                image_path = array_to_image_path(image)
                return run_showui(image, query) + (image_path,)

            submit_btn.click(
                on_submit,
                [imagebox, textbox],
                [output_img, output_coords, state_image_path],
            )

            clear_btn.click(
                lambda: (None, None, None, None, None),
                inputs=None,
                outputs=[imagebox, textbox, output_img, output_coords, state_image_path],  # Clear all outputs
                queue=False
            )

            regenerate_btn.click(
                lambda image, query, state_image_path: run_showui(image, query),
                [imagebox, textbox, state_image_path],
                [output_img, output_coords],
            )

            # Record vote actions without feedback messages
            vote_btn.click(
                lambda image_path, query, action_generated: handle_vote(
                    "upvote", image_path, query, action_generated
                ),
                inputs=[state_image_path, textbox, output_coords],
                outputs=[],
                queue=False
            )

            downvote_btn.click(
                lambda image_path, query, action_generated: handle_vote(
                    "downvote", image_path, query, action_generated
                ),
                inputs=[state_image_path, textbox, output_coords],
                outputs=[],
                queue=False
            )

            flag_btn.click(
                lambda image_path, query, action_generated: handle_vote(
                    "flag", image_path, query, action_generated
                ),
                inputs=[state_image_path, textbox, output_coords],
                outputs=[],
                queue=False
            )

    return demo
# Launch the app
if __name__ == "__main__":
    demo = build_demo(embed_mode=False)
    demo.queue(api_open=False).launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        debug=True,
    )
