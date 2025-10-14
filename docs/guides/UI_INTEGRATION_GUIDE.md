# 🎨 UI Integration Guide: User-Driven Autonomous Learning

## Overview

The frontend UI has been enhanced to provide full control and visibility over the autonomous learning system, clearly showing when the user provides input vs when the LLM generates prompts.

---

## 🎯 Key UI Components

### 1. **Loop Status Banner** (Top of LearningControl)

Shows real-time status with 3 states:

#### State 1: ⏳ Waiting for Initial Prompt (Yellow)
```
┌─────────────────────────────────────────────────────────┐
│ 🕐 ⏳ Waiting for Initial Prompt                        │
│                                                         │
│ ⏳ Waiting for initial user prompt from UI to start     │
│ autonomous learning.                                    │
└─────────────────────────────────────────────────────────┘
```
**What it means:** System is ready but needs YOUR first prompt to begin.

#### State 2: 🚀 Autonomous Learning Active (Green)
```
┌─────────────────────────────────────────────────────────┐
│ ⚡ 🚀 Autonomous Learning Active   [Iteration 42]       │
│                                                         │
│ ✅ Autonomous learning loop is active.                  │
│                                                         │
│ Initial Prompt:                                         │
│ "Teach me about Python basics and ML fundamentals..."  │
│ by user_123 • 10/12/2025, 8:30:00 PM                  │
└─────────────────────────────────────────────────────────┘
```
**What it means:** LLM is generating prompts automatically. Your initial prompt is shown for reference.

#### State 3: ⏸️ Learning Loop Stopped (Gray)
```
┌─────────────────────────────────────────────────────────┐
│ ⏸️ Learning Loop Stopped                                │
│                                                         │
│ ⏸️ Autonomous learning loop is stopped.                 │
└─────────────────────────────────────────────────────────┘
```
**What it means:** Loop was stopped. Submit new prompt to restart.

---

### 2. **Instructions Box** (Shows when waiting)

```
┌─────────────────────────────────────────────────────────┐
│ ℹ️ How User-Driven Learning Works:                      │
│                                                         │
│ 1. You submit the first prompt (below)                 │
│ 2. System processes it and starts autonomous loop      │
│ 3. LLM generates all subsequent prompts automatically   │
│ 4. System learns continuously without further input    │
└─────────────────────────────────────────────────────────┘
```

---

### 3. **Prompt Input Textarea**

#### When Waiting for Initial Prompt:
```
┌─────────────────────────────────────────────────────────┐
│  🎯 Enter your initial prompt here to kickstart        │ [Initial Prompt Required]
│  autonomous learning...                                │
│  (e.g., 'Teach me about Python basics and ML          │
│   fundamentals')                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### When Loop is Running:
```
┌─────────────────────────────────────────────────────────┐
│  Autonomous learning is active. LLM is generating      │ [Loop Running]
│  prompts automatically...                              │
│  (Disabled - LLM in control)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 4. **Submit Button**

#### When Waiting:
```
┌──────────────────────────────────────┐
│ 🚀 Submit Initial Prompt & Start Learning │
└──────────────────────────────────────┘
```

#### When Running:
```
┌──────────────────────────────────────┐
│ Restart with New Prompt (Disabled)  │
└──────────────────────────────────────┘
```

---

### 5. **Active Learning Info Box** (Shows when running)

```
┌─────────────────────────────────────────────────────────┐
│ ✓ Autonomous Learning Active - LLM is in Control       │
│                                                         │
│ Current Status:                What's Happening:       │
│ • Iteration #42               • System generates       │
│ • LLM generating prompts        new prompts (1/sec)    │
│   automatically               • Processes through      │
│ • Learning from each            curriculum             │
│   iteration                   • Continuous improvement │
│                                 via PPO & Q-learning   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete User Flow in UI

### Step 1: User Opens Dashboard

**UI Shows:**
- Yellow banner: "⏳ Waiting for Initial Prompt"
- Instructions box explaining 4-step process
- Textarea with helpful placeholder
- Big green button: "🚀 Submit Initial Prompt & Start Learning"

**User Action:** Types initial prompt

---

### Step 2: User Submits Initial Prompt

**UI Shows:**
- Button text changes: "Processing Initial Prompt..."
- Loading spinner appears

**Backend Action:** 
- Processes prompt via `/prompt/process`
- Automatically triggers `receive_initial_prompt()`
- Starts autonomous learning loop

---

### Step 3: Loop Starts (After ~1 second)

**UI Updates:**
- Banner turns green: "🚀 Autonomous Learning Active"
- Shows "Iteration 1", "Iteration 2", etc.
- Displays user's initial prompt
- Active learning info box appears
- Textarea becomes disabled (LLM is in control)
- Toast: "Initial prompt submitted! Autonomous learning will start automatically."

---

### Step 4: Autonomous Learning Runs

**UI Shows Real-Time:**
- Iteration counter increments (e.g., "Iteration 15")
- Learning History shows new entries (all source: "ai")
- Curriculum Progress updates
- Metrics increase
- All happening automatically!

**User Can:**
- Watch the progress
- Stop the loop
- View analytics
- See hallucination monitoring
- Check feedback

---

### Step 5: User Stops Loop (Optional)

**UI Shows:**
- Red "Stop Learning" button clicked
- Banner turns gray: "⏸️ Learning Loop Stopped"
- Textarea becomes enabled again
- Button changes: "Restart with New Prompt"

---

## 🎨 UI States Summary

| State | Banner Color | Button Text | Textarea | Iteration Badge |
|-------|-------------|-------------|----------|----------------|
| **Waiting** | 🟡 Yellow | 🚀 Submit Initial Prompt | Enabled | None |
| **Running** | 🟢 Green | Restart (Disabled) | Disabled | Iteration # |
| **Stopped** | ⚪ Gray | Restart with New Prompt | Enabled | None |

---

## 💻 Technical Implementation

### Frontend Changes

**File:** `frontend/src/components/dashboard/LearningControl.tsx`

#### Added State:
```typescript
const [loopStatus, setLoopStatus] = useState<LoopStatus | null>(null);

interface LoopStatus {
  is_running: boolean;
  waiting_for_initial_prompt: boolean;
  initial_prompt_received: boolean;
  current_iteration: number;
  initial_prompt_data?: {
    prompt_text: string;
    user_id: string;
    timestamp: number;
  };
  message: string;
}
```

#### Added Polling:
```typescript
useEffect(() => {
  fetchLoopStatus();
  const interval = setInterval(fetchLoopStatus, 3000); // Every 3 seconds
  return () => clearInterval(interval);
}, []);

const fetchLoopStatus = async () => {
  const response = await fetch('http://localhost:8082/learning/autonomous/status');
  const data = await response.json();
  setLoopStatus(data);
  setIsLearningActive(data.is_running);
};
```

#### Updated Submit Handler:
```typescript
const handleProcessPrompt = async () => {
  const response = await apiService.processUserPrompt(request, "user_dashboard");
  
  // Backend automatically triggers loop
  toast.success("Initial prompt submitted! Autonomous learning will start automatically.");
  
  // Refresh status after 1 second
  setTimeout(() => fetchLoopStatus(), 1000);
};
```

---

## 📊 API Integration

### Frontend → Backend Communication

#### 1. Submit Initial Prompt
```typescript
POST /prompt/process
Body: { "prompt_text": "Your initial prompt" }

Response: { 
  "response": "...",
  "action": 5,
  "metrics": {...}
}
```
**Trigger:** User clicks "Submit Initial Prompt"  
**Backend:** Automatically calls `receive_initial_prompt()` → starts loop

#### 2. Check Loop Status (Every 3 seconds)
```typescript
GET /learning/autonomous/status

Response: {
  "is_running": true,
  "waiting_for_initial_prompt": false,
  "initial_prompt_received": true,
  "current_iteration": 42,
  "initial_prompt_data": {
    "prompt_text": "Teach me Python",
    "user_id": "user_123",
    "timestamp": 1728765432.123
  },
  "message": "✅ Autonomous learning loop is active."
}
```
**Purpose:** Real-time UI updates

#### 3. Stop Loop
```typescript
POST /learning/stop

Response: { "success": true }
```
**Trigger:** User clicks "Stop Learning"

---

## 🎯 User vs LLM Prompts - Visual Indicators

### In Learning History Component

**User Prompts:**
```
┌─────────────────────────────────────────────────────────┐
│ Source: 👤 user    Prompt: "Teach me Python basics"    │
│ Response: "Python is a high-level..."                   │
│ Reward: 0.850                                           │
└─────────────────────────────────────────────────────────┘
```

**LLM Prompts (Autonomous):**
```
┌─────────────────────────────────────────────────────────┐
│ Source: 🤖 ai      Prompt: "Python data structures..."  │
│ Response: "Lists, tuples, dictionaries..."              │
│ Reward: 0.782                                           │
└─────────────────────────────────────────────────────────┘
```

**Key Difference:** 
- `source: "user"` = Your prompt
- `source: "ai"` = LLM-generated prompt

---

## 🧪 Testing the UI

### Test 1: Initial State
1. Open dashboard
2. **Expected:** Yellow banner "Waiting for Initial Prompt"
3. **Expected:** Instructions box visible
4. **Expected:** Button says "Submit Initial Prompt & Start Learning"

### Test 2: Submit Initial Prompt
1. Type: "Teach me machine learning"
2. Click submit button
3. **Expected:** Toast "Initial prompt submitted!"
4. **Expected:** Banner turns green within 1-2 seconds
5. **Expected:** Iteration counter starts: 1, 2, 3...

### Test 3: Autonomous Learning
1. Wait 10 seconds
2. **Expected:** Iteration count increases (10+ iterations)
3. **Expected:** Learning History shows new "ai" entries
4. **Expected:** Initial prompt displayed in banner
5. **Expected:** Active learning info box visible

### Test 4: Stop Loop
1. Click "Stop Learning" button
2. **Expected:** Banner turns gray "Loop Stopped"
3. **Expected:** Textarea becomes enabled
4. **Expected:** Iteration counter stops

---

## 📱 Responsive Design

All components are responsive:
- Mobile: Stacked layout
- Tablet: 2-column grid
- Desktop: Full width with proper spacing

---

## 🎨 Color Coding

| State | Color | Meaning |
|-------|-------|---------|
| 🟡 Yellow | `bg-yellow-50` | Waiting for input |
| 🟢 Green | `bg-green-50` | Active learning |
| ⚪ Gray | `bg-gray-50` | Stopped |
| 🔵 Blue | `bg-blue-50` | Informational |

---

## 🚀 Deployment Notes

### Required Frontend Dependencies
```json
{
  "@radix-ui/react-alert": "^1.0.0",
  "@radix-ui/react-badge": "^1.0.0",
  "lucide-react": "^0.263.1",
  "sonner": "^1.0.0"
}
```

### Environment Variables
```env
VITE_API_URL=http://localhost:8082
```

### Build & Run
```bash
cd frontend
npm install
npm run dev
```

---

## 📋 Summary

### ✅ User Provides:
1. **First prompt only** (via UI textarea)
2. Controls (start/stop buttons)
3. Observations (watches progress)

### ✅ LLM Provides:
1. **All subsequent prompts** (automatically)
2. Curriculum-aware generation
3. Progressive difficulty
4. Continuous learning

### ✅ UI Shows:
1. Clear waiting state
2. Real-time iteration count
3. Initial prompt reference
4. Source of each prompt (user vs ai)
5. Loop status at all times
6. Full control buttons

---

## 🎉 Benefits

1. ✅ **User knows exactly when to provide input** (only the first prompt)
2. ✅ **Clear visual feedback** on loop status
3. ✅ **Real-time updates** via polling (3 second intervals)
4. ✅ **Complete control** from UI (start/stop/restart)
5. ✅ **Source transparency** (user prompts vs LLM prompts)
6. ✅ **Context preservation** (initial prompt always visible)

---

*UI Integration Guide | Part of PS03 - Autonomous Multi-Objective Curriculum Learning Engine*
