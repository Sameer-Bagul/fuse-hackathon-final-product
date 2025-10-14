# 🎯 User-Driven Autonomous Learning Mode

## Overview

The autonomous learning system now requires an **initial user prompt** before starting. This gives users full control over when and how the autonomous learning begins, making the system more predictable and user-friendly.

---

## 🔄 How It Works Now

### Previous Behavior (Auto-Start)
```
1. Server starts: python app.py
2. Autonomous loop starts IMMEDIATELY
3. System generates prompts automatically
4. No user input required
```

### New Behavior (User-Driven)
```
1. Server starts: python app.py
2. System WAITS for user input ⏳
3. User submits initial prompt via UI
4. Autonomous loop STARTS with context
5. System continues learning autonomously
```

---

## 🚀 Getting Started

### Step 1: Start the Server

```bash
cd server-ml
python app.py
```

**Expected Output:**
```
✅ Service 'PersistenceService' initialization: SUCCESS
✅ Service 'MetaLearningService' initialization: SUCCESS
...
⏳ Autonomous learning loop will start after receiving initial user prompt from UI
💡 System is ready. Please submit an initial prompt through the UI to begin autonomous learning.
✅ All MVC components initialized successfully
INFO: Uvicorn running on http://0.0.0.0:8082
```

**Key Message:** 
> "⏳ Autonomous learning loop will start after receiving initial user prompt from UI"

The system is now **waiting** for your input!

---

### Step 2: Submit Initial Prompt

#### Option A: Via Frontend UI (Recommended)

1. Open the frontend dashboard
2. Navigate to the **prompt submission** interface
3. Enter your initial prompt (e.g., "Teach me about Python basics")
4. Click **Submit**

#### Option B: Via API (curl)

```bash
curl -X POST http://localhost:8082/prompt/process \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_text": "Explain the fundamentals of machine learning",
    "use_external_llm": false
  }'
```

#### Option C: Via API (Python)

```python
import requests

response = requests.post(
    "http://localhost:8082/prompt/process",
    json={
        "prompt_text": "What are the key concepts in reinforcement learning?",
        "use_external_llm": False
    }
)

print(response.json())
```

---

### Step 3: Autonomous Learning Begins!

Once you submit the initial prompt:

1. **System processes your prompt** ✅
2. **Autonomous loop starts automatically** 🚀
3. **Background iterations begin** (1 per second) 🔄
4. **System learns continuously** 📈

**Expected Logs:**
```
🎯 First user prompt detected! Triggering autonomous learning loop...
🎯 Received initial user prompt: 'Explain the fundamentals...' | User: user_123
🚀 Initial prompt received! Starting autonomous learning loop...
🚀 Starting autonomous learning loop with curriculum integration
📝 Initial prompt context: 'Explain the fundamentals of machine learning...' | User: user_123
✅ Autonomous learning loop started successfully
```

---

## 📊 Checking System Status

### Via API: Check Status

```bash
curl http://localhost:8082/learning/autonomous/status
```

#### Before Initial Prompt:
```json
{
  "is_running": false,
  "waiting_for_initial_prompt": true,
  "initial_prompt_received": false,
  "thread_alive": false,
  "current_iteration": 0,
  "total_rewards": 0,
  "average_reward": 0.0,
  "curriculum_skills_count": 0,
  "initial_prompt_data": null,
  "message": "⏳ Waiting for initial user prompt from UI to start autonomous learning."
}
```

#### After Initial Prompt:
```json
{
  "is_running": true,
  "waiting_for_initial_prompt": false,
  "initial_prompt_received": true,
  "thread_alive": true,
  "current_iteration": 15,
  "total_rewards": 15,
  "average_reward": 0.782,
  "curriculum_skills_count": 3,
  "initial_prompt_data": {
    "prompt_text": "Explain the fundamentals of machine learning and its key concepts",
    "user_id": "user_123",
    "timestamp": 1728765432.123
  },
  "message": "✅ Autonomous learning loop is active."
}
```

---

## 🎛️ Control the Learning Loop

### Start Loop (with initial prompt)

The loop starts **automatically** when you submit the first prompt via `/prompt/process`.

### Stop Loop

```bash
curl -X POST http://localhost:8082/learning/stop
```

### Restart Loop

```bash
# Stop first
curl -X POST http://localhost:8082/learning/stop

# Then start again (requires another initial prompt)
curl -X POST http://localhost:8082/learning/start
```

**Note:** After stopping, you may need to submit another initial prompt to restart.

---

## 🔧 Technical Details

### Code Changes

#### 1. LearningLoopService (`services/learning_loop_service.py`)

**New Properties:**
```python
self.initial_prompt_received = False       # Has initial prompt been received?
self.waiting_for_initial_prompt = True     # Is system waiting?
self.initial_prompt_data = None            # Store initial prompt info
```

**New Method:**
```python
def receive_initial_prompt(self, prompt_text: str, result: dict, user_id: str = None):
    """
    Receive initial prompt from user and trigger autonomous learning loop.
    This is the entry point for starting the autonomous learning system.
    """
    # Store prompt data
    # Mark as received
    # Start learning loop
```

**Updated Method:**
```python
def start_learning_loop(self):
    """Start the autonomous learning loop in a background thread"""
    
    # NEW CHECK: Verify initial prompt received
    if self.waiting_for_initial_prompt and not self.initial_prompt_received:
        logger.warning("⏳ Cannot start learning loop: Waiting for initial user prompt from UI")
        logger.info("💡 Please submit a prompt through the UI to begin autonomous learning")
        return
    
    # ... rest of existing code
```

**New Status Method:**
```python
def get_loop_status(self):
    """Get current status of the learning loop"""
    return {
        'is_running': self.is_running,
        'waiting_for_initial_prompt': self.waiting_for_initial_prompt,
        'initial_prompt_received': self.initial_prompt_received,
        'iterations': self.learning_stats['iterations'],
        'thread_alive': self.loop_thread.is_alive() if self.loop_thread else False,
        'initial_prompt_data': {...}  # Initial prompt details
    }
```

#### 2. App.py (`app.py`)

**Disabled Auto-Start (line ~278):**
```python
# DO NOT start autonomous learning loop automatically
# Wait for initial user prompt from UI
log_service_initialization("AutonomousLearningLoop", "info", {
    "message": "⏳ Autonomous learning loop will start after receiving initial user prompt from UI",
    "status": "waiting_for_user_input"
})
logger.info("💡 System is ready. Please submit an initial prompt through the UI to begin autonomous learning.")
```

**Trigger on First Prompt (line ~884):**
```python
# Process the user prompt
result = prompt_controller.handle_prompt(...)

# 🚀 IMPORTANT: Trigger autonomous learning loop with initial prompt
if learning_loop_service.waiting_for_initial_prompt and not learning_loop_service.initial_prompt_received:
    logger.info("🎯 First user prompt detected! Triggering autonomous learning loop...")
    learning_loop_service.receive_initial_prompt(request.prompt_text, result, user_id)
```

**Updated Status Endpoint:**
```python
@app.get("/learning/autonomous/status")
async def get_autonomous_learning_status():
    # Returns comprehensive status including waiting state
    loop_status = learning_loop_service.get_loop_status()
    # ...
```

---

## 🎯 Benefits of User-Driven Mode

### 1. **User Control** 🎮
- Users decide when learning starts
- More predictable behavior
- Better for demos and testing

### 2. **Context Awareness** 📝
- System learns from user's initial prompt
- Can adapt curriculum based on user interest
- Stores initial context for reference

### 3. **Resource Efficiency** ⚡
- No wasted iterations before user is ready
- System only runs when needed
- Better for development/testing

### 4. **Better UX** 😊
- Clear feedback to user
- Visible trigger point
- User feels in control

### 5. **Debugging** 🐛
- Easier to reproduce issues
- Controlled start point
- Better for testing

---

## 📚 Use Cases

### Use Case 1: Educational Demo
```
Scenario: Teaching students about autonomous learning

1. Start server
2. Show students the "waiting" status
3. Explain what will happen
4. Submit initial prompt together
5. Watch autonomous iterations
6. Observe curriculum progression
```

### Use Case 2: Focused Learning
```
Scenario: User wants system to learn about specific topic

1. User submits: "Teach me advanced Python techniques"
2. System processes prompt
3. Autonomous loop starts with Python focus
4. Curriculum adapts to Python-heavy tasks
5. System continuously improves Python skills
```

### Use Case 3: Testing & Development
```
Scenario: Developer testing new features

1. Start server (loop waits)
2. Set breakpoints
3. Submit test prompt
4. Debug first few iterations
5. Stop loop
6. Analyze results
7. Repeat with different prompt
```

---

## ⚠️ Important Notes

### Note 1: Initial Prompt is Mandatory
The system **will not start** without an initial prompt. This is by design.

### Note 2: First Prompt Only
Only the **first** prompt triggers the loop. Subsequent prompts don't restart it.

### Note 3: Restarting
To restart after stopping:
1. Stop the loop via API
2. The system resets the waiting flag
3. Submit a new initial prompt to restart

### Note 4: Manual Start
You can manually start without a prompt using:
```bash
curl -X POST http://localhost:8082/learning/start
```
But this bypasses the initial prompt requirement (not recommended).

---

## 🔄 Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│              User Starts Server                         │
│              $ python app.py                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           System Initializes All Services               │
│     waiting_for_initial_prompt = True                   │
│     initial_prompt_received = False                     │
│     is_running = False                                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         ⏳ System Waits for User Input                  │
│                                                         │
│   Status API returns: "Waiting for initial prompt"     │
│   Frontend shows: "Submit a prompt to begin"           │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ User submits prompt via UI
                         ▼
┌─────────────────────────────────────────────────────────┐
│        🎯 First Prompt Detected!                        │
│   POST /prompt/process receives prompt                  │
│   Processes prompt normally                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│     🚀 Trigger Autonomous Learning Loop                 │
│   learning_loop_service.receive_initial_prompt()        │
│   - Stores initial prompt data                          │
│   - Sets initial_prompt_received = True                 │
│   - Sets waiting_for_initial_prompt = False             │
│   - Calls start_learning_loop()                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│       ✅ Autonomous Loop Starts                         │
│   - Creates background thread                           │
│   - is_running = True                                   │
│   - Logs initial prompt context                         │
│   - Begins continuous iterations                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│     🔄 Autonomous Learning Loop Running                 │
│   while is_running:                                     │
│     1. Generate curriculum prompt                       │
│     2. Process through LLM                              │
│     3. Evaluate & get reward                            │
│     4. PPO store transition                             │
│     5. Update progress                                  │
│     6. Sleep 1 second                                   │
│     (Repeats forever until stopped)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

### Test 1: Verify Waiting State

```bash
# Start server
python app.py

# Check status (should be waiting)
curl http://localhost:8082/learning/autonomous/status

# Expected: waiting_for_initial_prompt = true
```

### Test 2: Trigger with Prompt

```bash
# Submit initial prompt
curl -X POST http://localhost:8082/prompt/process \
  -H "Content-Type: application/json" \
  -d '{"prompt_text": "Test prompt", "use_external_llm": false}'

# Check status (should be running)
curl http://localhost:8082/learning/autonomous/status

# Expected: is_running = true, initial_prompt_received = true
```

### Test 3: Verify Loop Runs

```bash
# Wait 10 seconds
sleep 10

# Check status
curl http://localhost:8082/learning/autonomous/status

# Expected: current_iteration > 0, total_rewards > 0
```

---

## 🎉 Summary

The autonomous learning system now follows a **user-driven** approach:

1. ✅ **Server starts** → System waits
2. ✅ **User submits prompt** → Loop starts
3. ✅ **Autonomous learning** → Continues forever
4. ✅ **Full control** → Users decide when to begin

This makes the system more **predictable**, **controllable**, and **user-friendly**!

---

*Last Updated: October 12, 2025*  
*Part of PS03 - Autonomous Multi-Objective Curriculum Learning Engine*
