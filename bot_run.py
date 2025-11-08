import os
import time
import torch
import discord
import cv2
import asyncio
import json
import numpy as np
import base64
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from discord.ext import commands
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gdrive_upload import upload_to_drive, delete_from_drive
import threading
from multiprocessing import Process

# ---------------- DISCORD SETUP ----------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

device = "cuda" if torch.cuda.is_available() else "cpu"
use_half = torch.cuda.is_available()
print(f"🧠 Using device: {device.upper()} {'(FP16)' if use_half else '(FP32)'}")

# ---------------- MODEL LOAD ----------------
MODEL_PATH = "RealESRGAN_x4plus.pth"
rrdbnet = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                  num_block=23, num_grow_ch=32, scale=4)
model = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=rrdbnet,
    tile=256,
    tile_pad=10,
    pre_pad=0,
    half=use_half,
    device=device
)

# ---------------- CREDIT SYSTEM ----------------
CREDITS_FILE = "credits.json"
def load_credits():
    if not os.path.exists(CREDITS_FILE):
        with open(CREDITS_FILE, "w") as f: json.dump({}, f)
    with open(CREDITS_FILE, "r") as f: return json.load(f)
def save_credits(data):
    with open(CREDITS_FILE, "w") as f: json.dump(data, f, indent=4)
def get_credits(user_id): return load_credits().get(str(user_id), 0)
def update_credits(user_id, delta):
    data = load_credits(); data[str(user_id)] = max(0, data.get(str(user_id), 0)+delta); save_credits(data)

# ---------------- CINEMATIC DEPTH ----------------
def apply_cinematic_depth(image):
    w,h=image.size
    image=ImageEnhance.Brightness(image).enhance(1.05)
    image=ImageEnhance.Contrast(image).enhance(1.12)
    image=ImageEnhance.Color(image).enhance(1.06)
    soft=image.filter(ImageFilter.GaussianBlur(2))
    image=Image.blend(image,soft,0.08)
    vign=Image.new("L",(w,h),0)
    for x in range(w):
        for y in range(h):
            d=((x-w/2)/(w/1.6))**2+((y-h/2)/(h/1.6))**2
            vign.putpixel((x,y),int(255*min(1,d**0.5)))
    vign=vign.filter(ImageFilter.GaussianBlur(200))
    vm=ImageOps.invert(vign)
    v3=Image.merge("RGB",(vm,vm,vm)).filter(ImageFilter.GaussianBlur(90))
    image=Image.blend(image,v3,0.15)
    arr=np.power(np.clip(np.array(image,dtype=np.float32)/255.0,0,1),0.93)*1.08
    return Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8))

# ---------------- UPSCALE FUNCTION ----------------
async def run_upscale(inter, input_path):
    start=time.time()
    msg=await inter.edit_original_response(content="🎞️ Hyper-Realistic Cinematic Upscale in progress… ⏳")
    try:
        img=cv2.imread(input_path,cv2.IMREAD_UNCHANGED)
        if img is None:
            await msg.edit(content="❌ Error: could not read image file."); return
        if len(img.shape)==3 and img.shape[2]==4:
            a=img[:,:,3]/255.0; rgb=img[:,:,:3]; white=np.ones_like(rgb)*255
            img=(rgb*a[:,:,None]+white*(1-a[:,:,None])).astype(np.uint8)
        h,w=img.shape[:2]; scale=5000/max(w,h); new_w,new_h=int(w*scale),int(h*scale)
        for p in [10,25,50,75]:
            await msg.edit(content=f"⏱️ Upscaling progress: {p}%"); await asyncio.sleep(1)
        output,_=model.enhance(img,outscale=scale)
        result=Image.fromarray(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
        result=apply_cinematic_depth(result)
        os.makedirs("results",exist_ok=True)
        fname=f"cinematic_ultrahd_{new_w}x{new_h}_{int(time.time())}.png"
        out_path=os.path.join("results",fname)
        result.save(out_path,format="PNG",dpi=(300,300),compress_level=0)
        size=os.path.getsize(out_path)/(1024*1024); dur=time.time()-start

        if size<=8:
            view=discord.ui.View()
            button=discord.ui.Button(label="⬇️ Download Image",style=discord.ButtonStyle.link,
                                     url="attachment://"+os.path.basename(out_path))
            view.add_item(button)
            await msg.edit(content=f"✅ **Cinematic Upscale Complete!**\n📐 {new_w}×{new_h}px @300 DPI\n💾 {size:.1f} MB • ⏱ {dur:.1f}s",view=view)
            await inter.followup.send(file=discord.File(out_path))
        else:
            await msg.edit(content=f"☁️ Uploading large file to Google Drive ({size:.1f} MB)… ⏳")
            link=upload_to_drive(out_path,fname)
            view=discord.ui.View()
            dl_button=discord.ui.Button(label="📥 Download from Drive",style=discord.ButtonStyle.link,url=link)
            view.add_item(dl_button)
            await msg.edit(content=f"✅ **Cinematic Upscale Complete!**\n📐 {new_w}×{new_h}px @300 DPI\n💾 {size:.1f} MB • ⏱ {dur:.1f}s",view=view)

            async def auto_delete():
                await asyncio.sleep(600)
                try:
                    delete_from_drive(link)
                    if os.path.exists(out_path): os.remove(out_path)
                    print(f"🧹 Deleted: {fname}")
                except Exception as e: print("Auto-delete failed:",e)
            bot.loop.create_task(auto_delete())

    except Exception as e:
        await msg.edit(content=f"❌ Upscale failed: {e}"); print("Error:",e)

# ---------------- SLASH COMMAND ----------------
@tree.command(name="upload-image",description="Upload an image for cinematic upscale")
async def upload_image(inter:discord.Interaction,file:discord.Attachment):
    uid=str(inter.user.id); c=get_credits(uid)
    if c<=0:
        await inter.response.send_message(f"❌ You have 0 credits, {inter.user.mention}."); return
    update_credits(uid,-1)
    os.makedirs("uploads",exist_ok=True)
    path=os.path.join("uploads",file.filename)
    await file.save(path)
    print(f"📥 Received {file.filename} from {inter.user}")
    view=discord.ui.View()
    btn=discord.ui.Button(label="🎬 Start Cinematic Upscale",style=discord.ButtonStyle.green)
    async def cb(btn_inter):
        await btn_inter.response.defer(thinking=True)
        await run_upscale(btn_inter,path)
    btn.callback=cb; view.add_item(btn)
    await inter.response.send_message(f"🖼️ Choose upscale option (Credits left: {c-1})",view=view)

# ---------------- READY ----------------
@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user}")
    try:
        synced=await tree.sync()
        print(f"🔗 Synced {len(synced)} commands.")
    except Exception as e: print("Sync error:",e)


# ---------------- FLASK API (for Gemini web app) ----------------
app = Flask(__name__)

@app.route("/api/upscale", methods=["POST"])
def upscale_endpoint():
    try:
        data = request.json.get("image")
        image_data = base64.b64decode(data.split(",")[1])
        img = Image.open(BytesIO(image_data)).convert("RGB")
        output, _ = model.enhance(np.array(img)[:, :, ::-1], outscale=4)
        result_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        result_img.save(buf, format="JPEG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({"upscaled_image": f"data:image/jpeg;base64,{result_b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- START BOTH ----------------
def run_flask():
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, use_reloader=False)

if __name__ == "__main__":
    # Start Flask in a separate process (no async/thread conflict)
    flask_process = Process(target=run_flask)
    flask_process.start()

    # Start Discord bot in main process
    print("🤖 Starting Discord bot...")
    bot.run("DISCORD_TOKEN")

    # When bot exits, also stop Flask
    flask_process.terminate()
