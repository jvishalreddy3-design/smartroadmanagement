import os
import base64
import certifi
import json
import tempfile
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Literal
import httpx 

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv

# Optional imports for PDF generation
try:
    from fpdf import FPDF
    import qrcode
except ImportError:
    print("Please install fpdf and qrcode: pip install fpdf qrcode Pillow httpx")

load_dotenv()

# --- Configuration & Secrets ---
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-production")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip('"').strip("'").strip()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DB_NAME = "smartcity_roads"

# --- Google OAuth Configuration ---
# Add these to your .env file:
#   GOOGLE_CLIENT_ID=your-google-client-id
#   GOOGLE_CLIENT_SECRET=your-google-client-secret
#   GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
#   FRONTEND_URL=http://localhost:5500   (or wherever index.html is served)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5500/frontend")

MONGODB_URL = os.getenv("MONGODB_URI", os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
if MONGODB_URL and MONGODB_URL.startswith('"') and MONGODB_URL.endswith('"'):
    MONGODB_URL = MONGODB_URL[1:-1]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="SmartRoad Professional API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Connection safely targets a single database
client = AsyncIOMotorClient(MONGODB_URL, tlsCAFile=certifi.where())
db = client[DB_NAME]
users_collection = db.users
complaints_collection = db.complaints
notifications_collection = db.notifications

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    display_name: Optional[str] = None   # Real name from Google OAuth; None for manual accounts
    role: str
    civic_score: Optional[int] = 0

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class TimelineEvent(BaseModel):
    status: str
    timestamp: datetime
    remark: Optional[str] = None

class ComplaintResponse(BaseModel):
    id: str
    user_id: Optional[str] = "Unknown"
    username: Optional[str] = "Unknown"
    description: Optional[str] = "No description"
    location: Optional[str] = "Unknown location"
    lat: Optional[float] = None
    lng: Optional[float] = None
    category: Optional[str] = "Other"
    priority: Optional[str] = "Medium"
    status: Optional[str] = "Pending"
    department: Optional[str] = None
    expected_resolution_date: Optional[datetime] = None
    admin_remark: Optional[str] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    image_base64: Optional[str] = None
    after_image_base64: Optional[str] = None
    upvotes: int = 0
    timeline: List[TimelineEvent] = []
    rating: Optional[int] = None
    ai_analysis: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    language: Literal["en", "hi", "te"] = "en"

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def fix_id(document):
    if not document: return document
    if "_id" in document:
        document["id"] = str(document["_id"])
        del document["_id"]
    if "created_at" not in document or not document["created_at"]:
        document["created_at"] = datetime.now(timezone.utc)
    return document

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError: raise credentials_exception
        
    user = await users_collection.find_one({"username": {"$regex": f"^{re.escape(token_data.username)}$", "$options": "i"}})
    if user is None: raise credentials_exception
    return user

async def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not enough privileges")
    return current_user

def cleanup_file(path: str):
    try:
        if os.path.exists(path): os.remove(path)
    except: pass

@app.on_event("startup")
async def seed_demo_data():
    print(f"\n{'='*50}")
    print(f"[STARTUP] Connected database: {DB_NAME}")
    print(f"[STARTUP] Groq AI Integration: {'Active' if GROQ_API_KEY else 'Missing API Key'}")
    print(f"{'='*50}\n")

@app.post("/api/chat")
async def ai_chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="AI Service is currently unavailable (API key missing).")
    
    current_date = datetime.now().strftime("%B %d, %Y")
    language_instructions = {
        "en": "Respond only in English.",
        "hi": "हिंदी में ही उत्तर दें। जब तक उपयोगकर्ता विशेष रूप से भाषा बदलने के लिए न कहे, अंग्रेज़ी में उत्तर न दें।",
        "te": "తెలుగులో మాత్రమే సమాధానం ఇవ్వండి. వినియోగదారు ప్రత్యేకంగా భాష మార్చమని అడిగితే తప్ప ఇంగ్లీష్‌లో సమాధానం ఇవ్వవద్దు.",
    }
    system_prompt = {
        "role": "system",
        "content": (
            f"You are the official SmartRoad Customer Care Assistant. Today's date is {current_date}. "
            "Be polite, friendly, and brief. Your job is to help users understand how to use the SmartRoad civic reporting platform. "
            "Features include: Register/Login, Submitting reports with GPS and photos, Tracking report status on the dashboard, "
            "Viewing the Public City Map, Upvoting issues, and Downloading PDF receipts. "
            "NEVER invent features, departments, or approval decisions. NEVER reveal personal data or backend technical details. "
            f"Language requirement: {language_instructions[req.language]}"
        )
    }
    
    messages = [system_prompt] + [{"role": m.role, "content": m.content} for m in req.messages[-10:]]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.5, "max_tokens": 300},
                timeout=15.0
            )
            resp.raise_for_status()
            data = resp.json()
            return {"reply": data["choices"][0]["message"]["content"]}
    except httpx.HTTPStatusError as e:
        try: err_msg = e.response.json().get("error", {}).get("message", e.response.text)
        except: err_msg = e.response.text
        
        if e.response.status_code == 401:
            raise HTTPException(status_code=500, detail="Invalid Groq API Key. Please verify your key in the .env file.")
        elif e.response.status_code == 429:
            raise HTTPException(status_code=500, detail="Groq API rate limit exceeded. Please wait a moment.")
        else:
            raise HTTPException(status_code=500, detail=f"Groq Error: {err_msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection Error: {str(e)}")

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="AI vision is disabled.")
    
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        mime_type = file.content_type or "image/jpeg"
        
        prompt = (
            "Analyze this user-uploaded photo for a civic road/infrastructure reporting app. "
            "Respond ONLY with a valid JSON object matching this exact structure: "
            "{\"issue_type\": \"Brief description of issue\", "
            "\"relevance\": \"Relevant\" or \"Unclear\" or \"Unrelated\", "
            "\"severity\": \"Low\" or \"Medium\" or \"High\" or \"Urgent\", "
            "\"reason\": \"One short sentence explaining why\", "
            "\"suggested_category\": \"Pothole\", \"Streetlight\", \"Garbage\", \"Water Leakage\", \"Drainage\", \"Traffic Signal\", \"Road Damage\", or \"Other\", "
            "\"suggested_priority\": \"Low\", \"Medium\", \"High\", or \"Urgent\"}"
        )

        payload = {
            "model": "llama-3.2-11b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 300
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
                timeout=20.0
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            if content.startswith("```json"): content = content[7:-3]
            elif content.startswith("```"): content = content[3:-3]
                
            return json.loads(content.strip())
            
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI Analysis failed to process the image.")

@app.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate):
    normalized_username = user.username.strip().lower()
    if await users_collection.find_one({"username": {"$regex": f"^{re.escape(normalized_username)}$", "$options": "i"}}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    role = "admin" if await users_collection.count_documents({}) == 0 else "citizen"
    
    result = await users_collection.insert_one({
        "username": user.username.strip(),
        "hashed_password": get_password_hash(user.password),
        "role": role,
        "civic_score": 0
    })
    return fix_id(await users_collection.find_one({"_id": result.inserted_id}))

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    clean_username = form_data.username.strip()
    
    user = await users_collection.find_one({
        "username": {"$regex": f"^{re.escape(clean_username)}$", "$options": "i"}
    })
    
    if not user:
        raise HTTPException(status_code=401, detail="Account not found. Please register first.")
        
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    
    token = create_access_token({"sub": user["username"], "role": user["role"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return fix_id(current_user)

@app.post("/complaints/", response_model=ComplaintResponse)
async def create_complaint(
    description: str = Form(...),
    location: str = Form(...),
    category: str = Form(...),
    priority: str = Form(...),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    image: UploadFile = File(None),
    ai_analysis: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    image_base64 = None
    if image:
        image_base64 = base64.b64encode(await image.read()).decode("utf-8")
        
    parsed_ai = None
    if ai_analysis:
        try: parsed_ai = json.loads(ai_analysis)
        except: pass

    now = datetime.now(timezone.utc)
    doc = {
        "user_id": str(current_user["_id"]),
        "username": current_user["username"],
        "description": description,
        "location": location,
        "lat": lat, "lng": lng,
        "category": category,
        "priority": priority,
        "status": "Pending",
        "created_at": now,
        "image_base64": image_base64,
        "upvotes": 0,
        "timeline": [{"status": "Pending", "timestamp": now, "remark": "Report submitted."}],
        "ai_analysis": parsed_ai
    }
    
    result = await complaints_collection.insert_one(doc)
    await users_collection.update_one({"_id": current_user["_id"]}, {"$inc": {"civic_score": 10}})
    
    return fix_id(await complaints_collection.find_one({"_id": result.inserted_id}))

@app.get("/complaints/public", response_model=List[ComplaintResponse])
async def get_public_complaints(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    query = {}
    if status: query["status"] = {"$in": status.split(",")} if "," in status else status
    if priority: query["priority"] = {"$in": priority.split(",")} if "," in priority else priority
    if category: query["category"] = category
    if start_date or end_date:
        date_query = {}
        if start_date:
            try: date_query["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except: pass
        if end_date:
            try: date_query["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except: pass
        if date_query: query["created_at"] = date_query
    
    limit = min(limit, 200)
    cursor = complaints_collection.find(query, {"image_base64": 0}).sort("created_at", -1).skip(skip).limit(limit)
    return [fix_id(c) for c in await cursor.to_list(length=limit)]

@app.get("/complaints/", response_model=List[ComplaintResponse])
async def read_all_complaints(
    current_user: dict = Depends(get_current_admin),
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    query = {}
    if status: query["status"] = status
    if priority: query["priority"] = priority
    if category: query["category"] = category
    if department:
        if department == "exists": query["department"] = {"$exists": True, "$ne": None, "$ne": ""}
        elif department == "null": query["department"] = {"$in": [None, ""]}
        else: query["department"] = department
    if start_date or end_date:
        date_query = {}
        if start_date:
            try: date_query["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except: pass
        if end_date:
            try: date_query["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except: pass
        if date_query: query["created_at"] = date_query
    
    limit = min(limit, 200)
    cursor = complaints_collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [fix_id(c) for c in await cursor.to_list(length=limit)]

@app.get("/my-complaints/", response_model=List[ComplaintResponse])
async def read_my_complaints(
    current_user: dict = Depends(get_current_user),
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    query = {"user_id": str(current_user["_id"])}
    if status: query["status"] = status
    if priority: query["priority"] = priority
    if category: query["category"] = category
    if start_date or end_date:
        date_query = {}
        if start_date:
            try: date_query["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except: pass
        if end_date:
            try: date_query["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except: pass
        if date_query: query["created_at"] = date_query
    
    limit = min(limit, 200)
    cursor = complaints_collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [fix_id(c) for c in await cursor.to_list(length=limit)]

@app.put("/complaints/{id}/status", response_model=ComplaintResponse)
async def update_status(
    id: str, 
    status: str = Form(...),
    priority: str = Form(None),
    department: str = Form(None),
    eta: str = Form(None),
    admin_remark: str = Form(None),
    after_image: UploadFile = File(None),
    current_user: dict = Depends(get_current_admin)
):
    try:
        complaint = await complaints_collection.find_one({"_id": ObjectId(id)})
        if not complaint: raise HTTPException(status_code=404)
            
        now = datetime.now(timezone.utc)
        update_data = {"status": status}
        changes = []
        
        if complaint.get("status") != status: changes.append(f"Status changed to {status}")
        if priority: 
            update_data["priority"] = priority
            if complaint.get("priority") != priority: changes.append(f"Priority changed to {priority}")
        if department: 
            update_data["department"] = department
            if complaint.get("department") != department: changes.append(f"Department assigned to {department}")
        if eta: 
            parsed_eta = datetime.fromisoformat(eta.replace('Z', '+00:00'))
            update_data["expected_resolution_date"] = parsed_eta
            if not complaint.get("expected_resolution_date") or complaint.get("expected_resolution_date").date() != parsed_eta.date():
                changes.append(f"ETA updated to {parsed_eta.date()}")
        if status == "Resolved" and not complaint.get("resolved_at"): 
            update_data["resolved_at"] = now
            changes.append("Report resolved")
        if after_image and after_image.filename:
            update_data["after_image_base64"] = base64.b64encode(await after_image.read()).decode("utf-8")
            changes.append("After-repair image uploaded")
            
        audit_trail = " | ".join(changes) if changes else "Report updated"
        final_remark = f"{audit_trail}. Official Note: {admin_remark}" if admin_remark else audit_trail
            
        timeline = complaint.get("timeline", [])
        timeline.append({"status": status, "timestamp": now, "remark": final_remark})
        update_data["timeline"] = timeline
        
        await complaints_collection.update_one({"_id": ObjectId(id)}, {"$set": update_data})
        
        if complaint.get("status") != status:
            await notifications_collection.insert_one({
                "user_id": complaint["user_id"],
                "title": f"Report Status: {status}",
                "message": f"Your report ({complaint.get('category', 'Issue')}) is now {status}. {admin_remark or ''}",
                "status": status,
                "is_read": False,
                "created_at": now
            })
        
        return fix_id(await complaints_collection.find_one({"_id": ObjectId(id)}))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update report")

@app.post("/complaints/{id}/upvote")
async def upvote_complaint(id: str, current_user: dict = Depends(get_current_user)):
    await complaints_collection.update_one({"_id": ObjectId(id)}, {"$inc": {"upvotes": 1}})
    return {"status": "success"}

@app.get("/notifications/")
async def get_notifications(current_user: dict = Depends(get_current_user)):
    cursor = notifications_collection.find({"user_id": str(current_user["_id"])}).sort("created_at", -1).limit(20)
    return [fix_id(n) for n in await cursor.to_list(length=20)]

@app.put("/notifications/{id}/read")
async def mark_notification_read(id: str, current_user: dict = Depends(get_current_user)):
    await notifications_collection.update_one({"_id": ObjectId(id), "user_id": str(current_user["_id"])}, {"$set": {"is_read": True}})
    return {"status": "success"}

@app.get("/complaints/{id}/pdf")
async def download_pdf(id: str, bg_tasks: BackgroundTasks):
    try:
        complaint = await complaints_collection.find_one({"_id": ObjectId(id)})
        if not complaint: raise HTTPException(status_code=404, detail="Complaint not found")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 15, "SmartRoad Official Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_text_color(0, 0, 0)
        details = [
            ("Complaint ID:", str(complaint.get("_id"))),
            ("Reporter:", complaint.get("username", "Unknown")),
            ("Category:", complaint.get("category", "N/A")),
            ("Priority:", complaint.get("priority", "N/A")),
            ("Status:", complaint.get("status", "Unknown")),
            ("Department:", complaint.get("department", "Not Assigned")),
            ("Location:", complaint.get("location", "N/A")),
            ("Coordinates:", f"{complaint.get('lat')}, {complaint.get('lng')}" if complaint.get('lat') else "N/A"),
            ("Report Date:", complaint.get("created_at").strftime('%Y-%m-%d %H:%M') if complaint.get("created_at") else "N/A"),
        ]
        
        for k, v in details:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(40, 8, k)
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8, str(v), ln=True)
            
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, "Description:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, complaint.get("description", ""))
        
        qr = qrcode.QRCode(version=1, box_size=4, border=2)
        qr.add_data(f"SmartRoad ID: {str(complaint['_id'])}")
        qr.make(fit=True)
        fd_qr, path_qr = tempfile.mkstemp(suffix=".png")
        os.close(fd_qr)
        qr.make_image(fill_color="black", back_color="white").save(path_qr)
        
        pdf.ln(10)
        pdf.image(path_qr, w=30)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, "Scan to verify report authenticity.", ln=True)
        
        fd_pdf, path_pdf = tempfile.mkstemp(suffix=".pdf")
        os.close(fd_pdf)
        pdf.output(path_pdf)
        
        bg_tasks.add_task(cleanup_file, path_qr)
        bg_tasks.add_task(cleanup_file, path_pdf)
        
        return FileResponse(path_pdf, filename=f"SmartRoad_{id}.pdf", media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Google OAuth Endpoints
# ---------------------------------------------------------------------------

@app.get("/auth/google/url")
async def google_login_url():
    """Step 1 – Return the Google OAuth URL to the frontend."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth is not configured. Add GOOGLE_CLIENT_ID to your .env file."
        )
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",       # always show account picker
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return {"url": auth_url}


@app.get("/auth/google/callback")
async def google_callback(code: str = None, error: str = None, state: str = None):
    """Step 2 – Google redirects here with ?code=… after the user consents."""

    # Handle user-cancelled or denied login
    if error or not code:
        reason = error or "access_denied"
        return RedirectResponse(
            f"{FRONTEND_URL}/#google_error={urllib.parse.quote(reason)}"
        )

    if not GOOGLE_CLIENT_SECRET:
        return RedirectResponse(
            f"{FRONTEND_URL}/#google_error=server_misconfigured"
        )

    # --- Exchange authorisation code for tokens ---
    try:
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                }
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()
            access_token = token_data.get("access_token")

            # --- Fetch user profile ---
            user_info_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_info_resp.raise_for_status()
            user_info = user_info_resp.json()

    except Exception as e:
        return RedirectResponse(f"{FRONTEND_URL}/#google_error=token_exchange_failed")

    email = user_info.get("email")
    if not email:
        return RedirectResponse(f"{FRONTEND_URL}/#google_error=no_email_provided")

    # Google provides the user's real full name — use it as the display name
    google_display_name = user_info.get("name", "").strip()

    # --- Create or fetch user in our DB ---
    # Use the portion of the email before @ as the internal username (unique identifier)
    username = email.split("@")[0]
    
    # Check if user already exists
    user = await users_collection.find_one({
        "username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}
    })
    
    if not user:
        # Create a new user account linked to Google
        role = "admin" if await users_collection.count_documents({}) == 0 else "citizen"
        new_user = {
            "username": username,
            "display_name": google_display_name,   # Real name shown in the UI
            "email": email,
            "hashed_password": get_password_hash(email), # Dummy hash, not used for oauth login
            "role": role,
            "civic_score": 0,
            "oauth_provider": "google"
        }
        result = await users_collection.insert_one(new_user)
        user = await users_collection.find_one({"_id": result.inserted_id})
    elif google_display_name:
        # Always keep the display_name fresh (user may have changed their Google name)
        await users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"display_name": google_display_name}}
        )
        user["display_name"] = google_display_name

    # --- Generate App JWT Token ---
    app_token = create_access_token(
        {"sub": user["username"], "role": user["role"]}, 
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # Redirect back to the frontend with the token in the URL fragment
    return RedirectResponse(f"{FRONTEND_URL}/#google_token={app_token}")