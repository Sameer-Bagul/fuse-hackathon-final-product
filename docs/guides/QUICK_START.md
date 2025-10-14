# 🎯 Quick Start Guide: User-Driven Autonomous Learning

## ⚡ TL;DR

**Before:** Autonomous loop starts automatically when server starts ❌  
**Now:** Autonomous loop waits for YOUR first prompt ✅

---

## 🚀 How to Use (3 Simple Steps)

### Step 1: Start the Server

```bash
cd server-ml
python app.py
```

**What you'll see:**
```
✅ All MVC components initialized successfully
⏳ Autonomous learning loop will start after receiving initial user prompt from UI
💡 System is ready. Please submit an initial prompt through the UI to begin autonomous learning.
INFO: Uvicorn running on http://0.0.0.0:8082
```

**Status:** 🟡 **WAITING** for your input

---

### Step 2: Submit Your First Prompt

#### Option A: Via Frontend (Recommended)
1. Open your browser
2. Go to the dashboard
3. Type your prompt (e.g., "Teach me Python")
4. Click Submit

#### Option B: Via API
```bash
curl -X POST http://localhost:8082/prompt/process \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_text": "Teach me about machine learning",
    "use_external_llm": false
  }'
```

**What happens:**
```
🎯 First user prompt detected! Triggering autonomous learning loop...
🚀 Initial prompt received! Starting autonomous learning loop...
✅ Autonomous learning loop started successfully
```

**Status:** 🟢 **RUNNING** autonomously

---

### Step 3: Watch It Learn!

The system now runs **autonomously** in the background:

```
📝 Iteration 1: Generated prompt for skill 'python_basics'
✅ Iteration 1 completed | Reward: 0.782 | Time: 0.12s

📝 Iteration 2: Generated prompt for skill 'ml_concepts'  
✅ Iteration 2 completed | Reward: 0.845 | Time: 0.10s

📝 Iteration 3: Generated prompt for skill 'data_structures'
✅ Iteration 3 completed | Reward: 0.791 | Time: 0.11s

... continues forever ...
```

---

## 📊 Check Status Anytime

```bash
curl http://localhost:8082/learning/autonomous/status
```

### Before Initial Prompt:
```json
{
  "waiting_for_initial_prompt": true,
  "message": "⏳ Waiting for initial user prompt from UI to start autonomous learning."
}
```

### After Initial Prompt:
```json
{
  "is_running": true,
  "waiting_for_initial_prompt": false,
  "current_iteration": 42,
  "message": "✅ Autonomous learning loop is active."
}
```

---

## 🎮 Control Commands

### Check Status
```bash
curl http://localhost:8082/learning/autonomous/status
```

### Stop Learning
```bash
curl -X POST http://localhost:8082/learning/stop
```

### Start Again (requires new initial prompt)
```bash
# Submit another prompt via /prompt/process
curl -X POST http://localhost:8082/prompt/process \
  -H "Content-Type: application/json" \
  -d '{"prompt_text": "New topic", "use_external_llm": false}'
```

---

## 🎨 Visual Flow

```
┌─────────────────────┐
│  Start Server       │
│  $ python app.py    │
└──────────┬──────────┘
           │
           ▼
    ⏳ WAITING STATE
    (Nothing happens)
           │
           │  User submits
           │  first prompt
           ▼
┌──────────────────────┐
│  Process Prompt      │
│  "Teach me Python"   │
└──────────┬───────────┘
           │
           ▼
    🚀 LOOP STARTS!
    (Automatically)
           │
           ▼
┌──────────────────────┐
│  Autonomous Learning │
│  Iteration 1, 2, 3...│
│  (Forever)           │
└──────────────────────┘
```

---

## ❓ FAQ

### Q: What happens if I don't submit a prompt?
**A:** Nothing! The system waits patiently. No iterations, no learning, no resource usage.

### Q: Can I start the loop manually?
**A:** You can use `POST /learning/start`, but it will fail until you submit an initial prompt first.

### Q: Does every prompt restart the loop?
**A:** No! Only the FIRST prompt triggers the loop. After that, it runs continuously.

### Q: Can I change the initial prompt later?
**A:** The initial prompt is stored for context, but you can stop the loop and start again with a new prompt.

### Q: What if the loop is already running?
**A:** Subsequent prompts are processed normally but don't restart the loop.

---

## 💡 Example Session

```bash
# Terminal 1: Start server
$ python app.py
⏳ Waiting for initial user prompt...

# Terminal 2: Check status (waiting)
$ curl http://localhost:8082/learning/autonomous/status
{"waiting_for_initial_prompt": true, "is_running": false}

# Terminal 2: Submit initial prompt
$ curl -X POST http://localhost:8082/prompt/process \
  -H "Content-Type: application/json" \
  -d '{"prompt_text": "Teach me Python", "use_external_llm": false}'

# Terminal 1: See loop start!
🎯 First user prompt detected!
🚀 Starting autonomous learning loop...
✅ Loop started successfully
📝 Iteration 1: Generated prompt...

# Terminal 2: Check status (running)
$ curl http://localhost:8082/learning/autonomous/status
{"waiting_for_initial_prompt": false, "is_running": true, "current_iteration": 5}

# Watch it learn autonomously!
```

---

## 🎯 Key Points

1. ✅ **Server starts** → System **WAITS** (does nothing)
2. ✅ **You submit prompt** → Loop **STARTS automatically**
3. ✅ **Loop runs forever** → Autonomous learning continues
4. ✅ **Full control** → You decide when to begin

---

## 🔗 Learn More

- **Complete Guide:** `USER_DRIVEN_LEARNING.md`
- **Technical Details:** `IMPLEMENTATION_SUMMARY.md`
- **How Loop Works:** `AUTONOMOUS_LOOP_EXPLANATION.md`

---

*Quick Start Guide | Part of PS03 - Autonomous Multi-Objective Curriculum Learning Engine*
