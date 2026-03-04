import os
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from typing import List
from processor.pipeline import run_card_pipeline
from processor.zipper import standardize_and_zip
from fastapi.responses import FileResponse, JSONResponse, Response

app = FastAPI(title="CardProcessor-Pro")

# Serve static files for the UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/process")
async def process_images(files: List[UploadFile] = File(...)):
    """
    Receives multiple images, processes them (bg removal, crop, upscale, redact),
    and returns a ZIP file containing the processed outputs.
    """
    processed_images = []
    
    for file in files:
        try:
            img_bytes = await file.read()
            processed_img = run_card_pipeline(img_bytes)
            processed_images.append((file.filename, processed_img))
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue
            
    if not processed_images:
        return JSONResponse(status_code=400, content={"error": "No valid images could be processed."})
        
    try:
        zip_stream = standardize_and_zip(processed_images)
        return Response(
            content=zip_stream.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=processed_cards.zip"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to zip images: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
