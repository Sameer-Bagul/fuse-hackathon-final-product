# 🐛 Common Issues & Solutions

Complete troubleshooting guide for the Autonomous Learning System.

---

## 🔴 Critical Issues

### Issue 1: Tensor Size Mismatch Errors

**Symptoms:**
```
ERROR - The size of tensor a (32) must match the size of tensor b (X)
ERROR - The size of tensor a (34) must match the size of tensor b (X)
Processing failed: tensor dimension mismatch
```

**Root Cause:**
PPO (reinforcement learning) buffer growing too large, causing dimension mismatches.

**Solution:**
```bash
# Simply restart the server - fixes are already applied
cd server-ml
# Press Ctrl+C to stop
python app.py
```

**What was fixed:**
- Added buffer size limit (max 100 transitions)
- Added tensor shape safety checks
- Auto-trims buffer when it gets too large

**Expected after restart:**
- ✅ Success rate improves to 70-90%
- ✅ Errors drop to <10%
- ✅ Loop runs smoothly

---

### Issue 2: Learning Loop Not Starting

**Symptoms:**
```
⏳ Waiting for initial user prompt
Loop status: Waiting
Banner shows yellow "Waiting" status
```

**Root Cause:**
System is designed to wait for your first prompt before starting autonomous learning.

**Solution:**
1. Open the UI at `http://localhost:5173`
2. Go to **Learning Control** panel
3. Type an initial prompt (e.g., "Teach me Python")
4. Click "🚀 Submit Initial Prompt & Start Learning"
5. Banner should turn green and iterations start

**What to check:**
- ✅ Backend server is running (port 8082)
- ✅ Frontend is accessible (port 5173)
- ✅ Prompt textarea is enabled
- ✅ Submit button is clickable

---

### Issue 3: Status Endpoint Returns 500 Error

**Symptoms:**
```
GET /learning/autonomous/status -> 500
ERROR - 'LearningLoopService' object has no attribute 'get_stats'
```

**Root Cause:**
Old code calling non-existent method.

**Solution:**
Already fixed in latest code! Just restart server:
```bash
cd server-ml
python app.py
```

**What was fixed:**
Changed `get_stats()` to `get_learning_progress()` in status endpoint.

---

## ⚠️ Common Issues

### Issue 4: Cannot Connect to MongoDB

**Symptoms:**
```
ERROR - Failed to connect to MongoDB
ConnectionError: Unable to reach database
```

**Solution:**
1. Check your `.env` file in `server-ml/`:
   ```bash
   DATABASE_URL=mongodb+srv://username:password@cluster.mongodb.net/
   ```

2. Verify MongoDB Atlas:
   - Cluster is active
   - IP whitelist includes your IP (or use 0.0.0.0/0 for all)
   - Username/password are correct

3. Test connection:
   ```bash
   cd server-ml
   python -c "from pymongo import MongoClient; client = MongoClient('YOUR_URL'); print(client.list_database_names())"
   ```

---

### Issue 5: Port Already in Use

**Symptoms:**
```
ERROR - Port 8082 is already in use
OSError: [Errno 98] Address already in use
```

**Solution Option 1:** Kill existing process
```bash
# Find process using port 8082
lsof -i :8082
# Kill it
kill -9 <PID>
# Restart server
python app.py
```

**Solution Option 2:** Change port
```bash
# Edit server-ml/.env
API_PORT=8083

# Or run with different port
python app.py --port 8083
```

---

### Issue 6: Frontend Not Connecting to Backend

**Symptoms:**
```
Network Error
Failed to fetch
CORS error
```

**Solution:**
1. Check backend is running:
   ```bash
   curl http://localhost:8082/health
   ```

2. Check CORS settings in `server-ml/.env`:
   ```bash
   CORS_ORIGINS=http://localhost:3000,http://localhost:5173
   ```

3. Verify frontend API URL in `frontend/src/lib/api.ts`:
   ```typescript
   const baseURL = 'http://localhost:8082';
   ```

---

### Issue 7: Dependencies Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement
npm ERR! code ERESOLVE
```

**Solution for Python:**
```bash
cd server-ml
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Solution for Node:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
# Or use bun
bun install
```

---

### Issue 8: PyTorch/NumPy Import Errors

**Symptoms:**
```
ImportError: No module named 'torch'
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
cd server-ml
source venv/bin/activate

# Reinstall PyTorch (CPU version)
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Reinstall NumPy
pip install --upgrade numpy
```

---

## 🟡 Performance Issues

### Issue 9: Slow Iteration Speed

**Symptoms:**
- Iterations taking > 5 seconds each
- UI feels sluggish
- High CPU usage

**Solutions:**

1. **Reduce batch size** in `models/ppo_agent.py`:
   ```python
   mini_batch_size: int = 32  # Reduce from 64
   ```

2. **Increase update frequency** in `services/learning_loop_service.py`:
   ```python
   if iteration_count % 5 == 0:  # Update every 5 instead of 10
   ```

3. **Disable external LLM** in `.env`:
   ```bash
   EXTERNAL_LLM_ENABLED=false
   ```

---

### Issue 10: High Memory Usage

**Symptoms:**
- Server using > 2GB RAM
- System slowing down
- Out of memory errors

**Solutions:**

1. **Reduce buffer size** in `models/ppo_agent.py`:
   ```python
   self.max_buffer_size = 50  # Reduce from 100
   ```

2. **Clear logs** regularly:
   ```bash
   cd server-ml/logs
   > llm_learning.log
   ```

3. **Limit history** in MongoDB - keep only last 1000 interactions

---

## 🟢 Best Practices to Avoid Issues

### ✅ Do's:
1. **Always restart server** after code changes
2. **Check logs** in `server-ml/logs/` for errors
3. **Use virtual environment** for Python dependencies
4. **Keep MongoDB connection string** secure
5. **Monitor success rate** - should be 70-90%
6. **Submit clear initial prompts** with 3-5 concepts

### ❌ Don'ts:
1. **Don't run multiple servers** on same port
2. **Don't commit `.env` file** with secrets
3. **Don't skip virtual environment** setup
4. **Don't ignore yellow warnings** in logs
5. **Don't submit single-word prompts** (too vague)

---

## 📊 Health Check Checklist

Use this checklist to verify system is healthy:

```bash
# 1. Backend health
curl http://localhost:8082/health
# Should return: {"status": "healthy"}

# 2. Frontend accessible
curl http://localhost:5173
# Should return HTML

# 3. Database connected
# Check logs: "MongoDB connected successfully"

# 4. Loop status
curl http://localhost:8082/learning/autonomous/status
# Should return JSON with loop status

# 5. Check logs
tail -f server-ml/logs/llm_learning.log
# Should see iteration logs

# 6. Monitor UI
# Open browser, check all panels load
```

---

## 🔍 Debug Mode

Enable detailed logging:

```bash
# Edit server-ml/.env
LOG_LEVEL=DEBUG

# Restart server
python app.py

# Now check logs for detailed info
tail -f server-ml/logs/llm_learning.log
```

---

## 📞 Still Having Issues?

1. **Check logs**: `server-ml/logs/llm_learning.log`
2. **Review documentation**: [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)
3. **Check specific guides**:
   - [Tensor Errors](TENSOR_ERRORS.md)
   - [Loop Not Starting](LOOP_NOT_STARTING.md)
4. **Try fresh start**:
   ```bash
   # Backend
   cd server-ml
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   
   # Frontend
   cd frontend
   rm -rf node_modules
   npm install
   npm run dev
   ```

---

**Last Updated**: 2025-10-12  
**Status**: ✅ All known issues documented with solutions
