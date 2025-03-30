import os
import logging
import tempfile
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
import torch

# Initialize logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configuration
TOKEN = "7968419921:AAF0yfB5nET_2lDrxqNUOuBYiTHDKvDi9H4"

# Initialize models in background
async def initialize_models():
    global image_pipe, video_pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize image pipeline
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    ).to(device)
    
    # Initialize video pipeline
    video_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    video_pipe.enable_model_cpu_offload()

# Start command with async initialization
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Initializing AI models...")
    asyncio.create_task(initialize_models())
    await update.message.reply_text("Hi! I'm ready to create!\n"
                                   "/generate_image [prompt]\n"
                                   "/generate_video [prompt]")

# Image generation with progress indication
async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'image_pipe' not in globals():
        await update.message.reply_text("⏳ Models still loading, please wait...")
        return
    
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt.")
        return
    
    async with update.message._chat.send_action(action="upload_photo"):
        try:
            image = await asyncio.to_thread(
                image_pipe, prompt=prompt
            ).images[0]
            
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                image.save(temp_file.name)
                await update.message.reply_photo(photo=open(temp_file.name, 'rb'))
                
        except Exception as e:
            logging.error(f"Image error: {str(e)}")
            await update.message.reply_text("❌ Error generating image")

# Video generation with progress handling
async def generate_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'video_pipe' not in globals():
        await update.message.reply_text("⏳ Models still loading, please wait...")
        return
    
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt.")
        return
    
    async with update.message._chat.send_action(action="upload_video"):
        try:
            frames = await asyncio.to_thread(
                video_pipe, prompt=prompt, num_frames=24, decode_chunk_size=8
            ).frames[0]
            
            with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
                frames[0].save(
                    temp_file.name,
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0
                )
                await update.message.reply_video(video=open(temp_file.name, 'rb'))
                
        except Exception as e:
            logging.error(f"Video error: {str(e)}")
            await update.message.reply_text("❌ Error generating video")

def main():
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("generate_image", generate_image))
    application.add_handler(CommandHandler("generate_video", generate_video))
    
    application.run_polling()

if __name__ == "__main__":
    main()
