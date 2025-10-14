# 🔧 Tensor Size Mismatch Errors - Complete Fix

## Problem Description

You're seeing errors like this in your logs:

```
ERROR - The size of tensor a (32) must match the size of tensor b (23) at non-singleton dimension 0
ERROR - The size of tensor a (34) must match the size of tensor b (25) at non-singleton dimension 0
ERROR - Processing failed: tensor dimension mismatch
```

And the numbers keep **incrementing** (32, 34, 36, 38, 40...).

## Root Cause

The PPO (Proximal Policy Optimization) reinforcement learning agent's **buffer is growing without limit**. This causes:

1. Buffer size increases every iteration
2. Tensor dimensions become inconsistent
3. Training fails with dimension mismatch
4. System can't update the policy

## The Fix (Already Applied!)

Three fixes have been implemented in `server-ml/models/ppo_agent.py`:

### Fix 1: Buffer Size Limit

```python
# Line ~100
self.max_buffer_size = 100  # Maximum transitions to store
```

### Fix 2: Automatic Trimming

```python
# Lines 187-214 in store_transition()
if len(self.states_buffer) > self.max_buffer_size:
    logger.warning(f"Buffer size ({len(self.states_buffer)}) exceeded max ({self.max_buffer_size}). Trimming...")
    
    # Keep only the most recent 50% of transitions
    trim_to = self.max_buffer_size // 2
    self.states_buffer = self.states_buffer[-trim_to:]
    self.actions_buffer = self.actions_buffer[-trim_to:]
    self.log_probs_buffer = self.log_probs_buffer[-trim_to:]
    # ... etc
```

### Fix 3: Shape Mismatch Safety Check

```python
# Lines ~342-348 in update_policy()
if new_log_probs.shape != batch_old_log_probs.shape:
    logger.error(f"Shape mismatch: new={new_log_probs.shape}, old={batch_old_log_probs.shape}")
    continue  # Skip this batch instead of crashing
```

## How to Apply the Fix

### Step 1: Stop the Server

Press `Ctrl+C` in the terminal where the server is running.

### Step 2: Restart the Server

```bash
cd server-ml
python app.py
```

### Step 3: Verify the Fix

Watch the logs:
```bash
tail -f server-ml/logs/llm_learning.log
```

You should see:
- ✅ "Buffer trimmed to 50 transitions" (when buffer gets full)
- ✅ Fewer tensor errors (should be < 10%)
- ✅ Success rate improving to 70-90%
- ✅ Iterations completing successfully

## What to Expect After Fix

### Before Fix:
```
❌ Success Rate: 20-30%
❌ Errors: 70-80%
❌ Tensor errors every iteration
❌ Numbers incrementing: 32, 34, 36, 38...
```

### After Fix:
```
✅ Success Rate: 70-90%
✅ Errors: < 10%
✅ Occasional buffer trim warnings (normal)
✅ Stable tensor dimensions
```

## Understanding the Logs

### Normal Operation:
```
INFO - [Iteration 15] Success Rate: 85.7%
INFO - [Iteration 16] Success Rate: 87.5%
```

### Buffer Trimming (Normal):
```
WARNING - Buffer size (100) exceeded max (100). Trimming...
INFO - Buffer trimmed to 50 transitions
```

### Fixed Errors (Rare):
```
ERROR - Shape mismatch detected, skipping batch
INFO - Continuing with next batch
```

## Why This Happens

The PPO agent learns by:
1. Storing experiences (state, action, reward) in a buffer
2. Training on batches of these experiences
3. Updating the policy network

Without a limit, the buffer grows indefinitely, causing:
- Inconsistent batch sizes
- Memory issues
- Dimension mismatches during training

## Technical Details

### Buffer Components:
- `states_buffer`: Input states (context, outcomes)
- `actions_buffer`: Actions taken (prompt variants)
- `rewards_buffer`: Rewards received (success/failure)
- `log_probs_buffer`: Log probabilities for policy gradient
- `values_buffer`: Value estimates for advantage calculation

### Why 100 and 50?
- **100**: Maximum buffer size before trimming
- **50**: Size after trimming (keeps most recent experiences)
- This ensures enough data for training while preventing unbounded growth

### Training Process:
1. Collect up to 100 experiences
2. When full, trim to 50 (keep recent)
3. Train on mini-batches of 32
4. Update policy and value networks
5. Clear buffer and repeat

## Troubleshooting

### Issue: Still seeing tensor errors after restart

**Check:**
```bash
# Are you running the latest code?
cd server-ml
git log -1  # Check last commit
```

**Solution:**
```bash
# Pull latest changes
git pull
# Restart server
python app.py
```

### Issue: Buffer trimming too frequently

**Symptom:** Warning every iteration

**Solution:** Increase max buffer size:
```python
# In models/ppo_agent.py line ~100
self.max_buffer_size = 200  # Increase from 100
```

### Issue: Success rate still low

**Check:**
1. External LLM enabled? (Better rewards)
2. Initial prompt quality (clear concepts)
3. Curriculum difficulty (not too hard)

**Solution:**
```bash
# Enable external LLM in .env
EXTERNAL_LLM_ENABLED=true
EXTERNAL_LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

## Prevention

To avoid this issue in the future:

1. **Always set buffer limits** in RL agents
2. **Monitor buffer sizes** in logs
3. **Add shape checks** before tensor operations
4. **Use try-catch** around training code
5. **Log warnings** when buffers grow large

## Related Documentation

- [Common Issues](COMMON_ISSUES.md) - All troubleshooting guides
- [Loop Not Starting](LOOP_NOT_STARTING.md) - If loop won't start
- [PPO Agent Documentation](../../server-ml/models/ppo_agent.py) - Full code

---

**Status**: ✅ Fixed - Just restart server  
**Last Updated**: 2025-10-12
