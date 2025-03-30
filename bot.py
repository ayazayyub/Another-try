import os
import logging
import tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from diffusers import StableDiffusionPipeline, StableDiffusionVideoPipeline
import torch
from transformers import pipeline

# Initialize logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configuration
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
MODEL_NAME = "stabilityai/stable-diffusion-2-1"
TEXT_MODEL = "google/flan-t5-large"

# Initialize models
text_pipeline = pipeline("text2text-generation", model=TEXT_MODEL)
image_pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
video_pipe = StableDiffusionVideoPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
image_pipe = image_pipe.to(device)
video_pipe = video_pipe.to(device)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm your AI assistant. Commands:\n"
                                   "/answer [question] - Get answers\n"
                                   "/generate_image [prompt] - Create image\n"
                                   "/generate_video [prompt] - Create video")

async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a question.")
        return
    
    response = text_pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        temperature=0.7
    )
    
    await update.message.reply_text(response[0]['generated_text'])

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt.")
        return
    
    image = image_pipe(prompt).images[0]
    
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        image.save(temp_file.name)
        await update.message.reply_photo(photo=open(temp_file.name, 'rb'))

async def generate_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt.")
        return
    
    video_frames = video_pipe(prompt, height=320, width=576, num_frames=24).frames
    video_path = "output.mp4"
    
    # Convert frames to video using ffmpeg
    import subprocess
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, frame in enumerate(video_frames):
            frame.save(f"{tmp_dir}/frame_{i:04d}.png")
        
        subprocess.call([
            'ffmpeg', '-y', '-framerate', '8',
            '-i', f'{tmp_dir}/frame_%04d.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            video_path
        ])
    
    await update.message.reply_video(video=open(video_path, 'rb'))

def main():
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("answer", answer))
    application.add_handler(CommandHandler("generate_image", generate_image))
    application.add_handler(CommandHandler("generate_video", generate_video))
    
    application.run_polling()

if __name__ == "__main__":
    main()
