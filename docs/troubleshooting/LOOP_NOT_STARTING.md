# 🚀 Learning Loop Not Starting - Complete Fix

## Problem Description

The autonomous learning loop is stuck in "waiting" state:

### Symptoms:
- Banner shows: **⏳ Waiting for initial user prompt**
- Status: **Waiting** (yellow)
- Iteration count stays at **0** or **1**
- Even after waiting minutes/hours, nothing happens
- Logs show: "Loop waiting for initial prompt"

## Root Cause

The system is **designed** to wait for your first prompt before starting autonomous learning. This is intentional!

**Why?** The system needs to know what to teach you before it can start learning how to teach.

## The Fix (Already Applied!)

A critical trigger was missing. The UI calls `/prompt/handle` endpoint, but the trigger to start the loop was only in `/prompt/process`.

**Fixed in `server-ml/app.py` (line ~587):**

```python
@app.post("/prompt/handle")
async def handle_user_prompt(request: ProcessPromptRequest):
    # ... existing code ...
    
    # CRITICAL FIX: Trigger loop start
    receive_initial_prompt()  # <-- This was missing!
    
    return {
        "success": True,
        "message": "Prompt received and processed"
    }
```

## How to Start the Loop

### Step 1: Ensure Server is Running

```bash
cd server-ml
python app.py
```

Wait for:
```
✅ MongoDB connected successfully
✅ Server running on port 8082
⏳ Waiting for initial user prompt
```

### Step 2: Open the UI

Open your browser to:
```
http://localhost:5173
```

### Step 3: Submit Initial Prompt

1. Find the **Learning Control** panel
2. Look for the prompt textarea (large text box)
3. Type your initial learning goal

**Example prompts:**
```
Teach me Python programming fundamentals
Help me learn web development with React
I want to understand machine learning basics
Show me how to build REST APIs with FastAPI
```

4. Click: **🚀 Submit Initial Prompt & Start Learning**

### Step 4: Verify Loop Started

Within 5-10 seconds, you should see:

**UI Changes:**
- ✅ Banner turns **green**: "Learning loop running"
- ✅ Status changes to: **Running**
- ✅ Iteration count starts incrementing: 1, 2, 3...
- ✅ Success rate appears: ~70-90%

**Backend Logs:**
```bash
tail -f server-ml/logs/llm_learning.log

# You should see:
INFO - Initial prompt received: "Teach me Python..."
INFO - Curriculum generated with 5 tasks
INFO - Learning loop started
INFO - [Iteration 1] Success Rate: 75.0%
INFO - [Iteration 2] Success Rate: 80.0%
```

## Understanding the Process

### 1. Initial State (Before Prompt)
```
User → System is idle
       Curriculum is empty
       Loop is waiting
```

### 2. After Submitting Prompt
```
User → Submits: "Teach me Python"
       ↓
System → Generates curriculum:
         - Task 1: Variables and data types
         - Task 2: Functions and loops
         - Task 3: Classes and OOP
         - Task 4: File handling
         - Task 5: Error handling
       ↓
Loop → Starts autonomous learning
```

### 3. Autonomous Learning Begins
```
Loop → Generates learning materials
     → Gets feedback (simulated or real)
     → Adjusts approach with RL
     → Repeats every 3-5 seconds
```

## Troubleshooting

### Issue 1: Prompt submission but loop still waiting

**Possible causes:**
1. Server not restarted after fix
2. Frontend not connected to backend
3. MongoDB connection issue

**Solutions:**

```bash
# 1. Restart server (loads the fix)
cd server-ml
# Press Ctrl+C
python app.py

# 2. Check backend is accessible
curl http://localhost:8082/health
# Should return: {"status": "healthy"}

# 3. Check MongoDB connection in logs
grep "MongoDB" server-ml/logs/llm_learning.log
# Should see: "MongoDB connected successfully"

# 4. Test prompt submission manually
curl -X POST http://localhost:8082/prompt/handle \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Teach me Python", "user_id": "test"}'
```

### Issue 2: Loop starts then immediately stops

**Check logs for errors:**
```bash
tail -f server-ml/logs/llm_learning.log | grep ERROR
```

**Common causes:**
- Tensor errors (see [TENSOR_ERRORS.md](TENSOR_ERRORS.md))
- MongoDB disconnection
- Missing environment variables

**Solution:**
```bash
# Check .env file
cat server-ml/.env

# Required variables:
# DATABASE_URL=mongodb+srv://...
# API_PORT=8082
# EXTERNAL_LLM_ENABLED=true
```

### Issue 3: Frontend shows "Network Error"

**Cause:** Frontend can't reach backend

**Solution:**
```bash
# 1. Verify backend port
netstat -tuln | grep 8082
# Should show: LISTEN on 8082

# 2. Check CORS settings in .env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# 3. Restart both services
# Terminal 1:
cd server-ml && python app.py

# Terminal 2:
cd frontend && npm run dev
```

### Issue 4: Button is disabled/grayed out

**Cause:** Loop already running OR textarea empty

**Solution:**
1. Check if loop is already running (green banner)
2. If yes, you're good! Loop is working
3. If no, ensure textarea has text (minimum 10 characters)
4. Refresh page and try again

## Good Initial Prompts

### ✅ Good Examples:
```
"Teach me Python programming fundamentals including variables, functions, and classes"
"Help me learn web development with HTML, CSS, and JavaScript"
"I want to understand machine learning algorithms like regression and classification"
"Show me how to build REST APIs with FastAPI and connect to databases"
```

### ❌ Bad Examples:
```
"Python" (too vague)
"Help" (no context)
"Teach me everything" (too broad)
"Code" (unclear goal)
```

### What Makes a Good Prompt?
1. **Specific topic** (Python, web dev, ML)
2. **Clear scope** (fundamentals, basics, specific concepts)
3. **3-5 concepts** mentioned (variables, functions, classes)
4. **Actionable** (teach, help, show)
5. **10+ words** (enough context)

## Advanced: Manual Loop Control

You can also control the loop via API:

### Start Loop:
```bash
curl -X POST http://localhost:8082/learning/autonomous/start
```

### Stop Loop:
```bash
curl -X POST http://localhost:8082/learning/autonomous/stop
```

### Check Status:
```bash
curl http://localhost:8082/learning/autonomous/status
```

## Expected Timeline

After submitting prompt:

```
T+0s:   Prompt submitted
T+1s:   Curriculum generated
T+2s:   Loop started
T+3s:   First iteration begins
T+6s:   Iteration 1 complete
T+9s:   Iteration 2 complete
T+12s:  Iteration 3 complete
...continues every 3-5 seconds
```

## Success Indicators

You know the loop is working when:

1. ✅ **Green banner** in UI
2. ✅ **Iteration count** incrementing
3. ✅ **Success rate** 70-90%
4. ✅ **Logs showing iterations** every few seconds
5. ✅ **Curriculum tasks** appearing in UI
6. ✅ **Feedback being generated**
7. ✅ **Rewards being calculated**

## Prevention

To avoid this issue in the future:

1. Always submit initial prompt before expecting autonomous learning
2. Don't submit empty or single-word prompts
3. Wait 5-10 seconds after submission
4. Check logs if loop doesn't start
5. Restart server after code changes

## Related Documentation

- [Common Issues](COMMON_ISSUES.md) - All troubleshooting
- [Initial Prompt Examples](../guides/INITIAL_PROMPT_EXAMPLES.md) - Prompt templates
- [Quick Start Guide](../guides/QUICK_START.md) - Setup instructions

---

**Status**: ✅ Fixed - Just restart server and submit prompt  
**Last Updated**: 2025-10-12
