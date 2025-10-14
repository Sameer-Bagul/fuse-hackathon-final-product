# LLM Learning System API Reference

## Overview

The LLM Learning System API provides RESTful endpoints for managing and analyzing an AI learning system built with MVC architecture. This API enables interaction with the learning model, task scheduling, evaluation, and data visualization.

## Base URL
```
http://localhost:8081
```

## Authentication
Currently, no authentication is required. In production, consider implementing JWT or API key authentication.

## Response Format
All responses are in JSON format. Successful responses include relevant data, while errors include a `detail` field with error information.

## Error Codes
- `400` - Bad Request (invalid input)
- `404` - Not Found (endpoint doesn't exist)
- `500` - Internal Server Error (server-side issues)

---

## Endpoints

### Health Check

#### GET `/health`
Check if the API is running and get system information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "version": "1.0.0",
  "environment": "development"
}
```

---

### Prompt Management

#### POST `/prompt/handle`
Process a user prompt and get the LLM's response.

**Request Body:**
```json
{
  "prompt_text": "Explain machine learning"
}
```

**Response:**
```json
{
  "response": "Machine learning is a subset of AI...",
  "action": 2
}
```

**Error Responses:**
- `400` - Empty or invalid prompt text
- `500` - LLM processing failed

#### GET `/prompt/metrics`
Retrieve learning metrics and performance statistics.

**Response:**
```json
{
  "success_rate": 0.85,
  "pattern_frequency": {
    "machine": 5,
    "learning": 8,
    "data": 3
  },
  "total_interactions": 42
}
```

---

### Task Scheduling

#### POST `/schedule`
Generate and prioritize learning tasks based on prompts.

**Request Body:**
```json
{
  "num_tasks": 5,
  "prompt_text": "Teach me about neural networks"
}
```

**Response:**
```json
{
  "tasks": [
    [0.8, 0.6, 0.9, 0.4, 0.7],
    [0.7, 0.8, 0.5, 0.9, 0.6],
    [0.9, 0.7, 0.8, 0.5, 0.6]
  ]
}
```

**Notes:**
- Tasks are represented as arrays of priority scores
- Higher scores indicate higher priority
- Tasks are automatically sorted by overall priority

---

### Evaluation

#### POST `/evaluate`
Evaluate the LLM's performance on learning tasks.

**Request Body:**
```json
{
  "prompt_text": "What is reinforcement learning?",
  "num_episodes": 100
}
```

**Response:**
```json
{
  "average_reward": 0.734
}
```

**Parameters:**
- `prompt_text` (optional): Specific prompt for evaluation
- `num_episodes`: Number of evaluation episodes to run

---

### Visualization

#### GET `/visualize/chartjs`
Get learning data formatted for Chart.js visualizations.

**Response:**
```json
{
  "chart_data": "{\"learning_curve\": {\"labels\": [...], \"datasets\": [...]}}"
}
```

**Notes:**
- Returns JSON string containing Chart.js configuration
- Includes learning curves, success rates, and pattern analysis
- Ready to use with Chart.js library in web frontends

#### GET `/visualize/learning_progress`
Get comprehensive learning progress data.

**Response:**
```json
{
  "metrics": {
    "success_rate": 0.85,
    "pattern_frequency": {...},
    "total_interactions": 42
  },
  "insights": [
    "🎉 Significant learning progress detected! Performance improved by 15.3%",
    "📊 Most common prompt pattern: 'learning' (appears 8 times)"
  ],
  "timestamp": 1703123456.789,
  "data_quality": "good"
}
```

#### POST `/visualize/generate_report`
Generate a comprehensive learning report.

**Response:**
```json
{
  "summary": {
    "total_episodes": 150,
    "average_reward": 0.723,
    "max_reward": 0.95,
    "min_reward": 0.12,
    "total_interactions": 42,
    "success_rate": 0.85
  },
  "charts": {
    "learning_curve": "{\"learning_curve\": {...}}"
  },
  "insights": [
    "✅ High success rate indicates good learning performance",
    "📈 Learning curve shows consistent improvement"
  ]
}
```

---

## Data Models

### PromptRequest
```python
{
  "prompt_text": str  # The user's prompt (required)
}
```

### PromptResponse
```python
{
  "response": str,    # LLM's response text
  "action": int       # Action taken by the LLM
}
```

### MetricsResponse
```python
{
  "success_rate": float,           # Overall success rate (0-1)
  "pattern_frequency": dict,       # Word frequency analysis
  "total_interactions": int        # Total prompts processed
}
```

### ScheduleRequest
```python
{
  "num_tasks": int,        # Number of tasks to generate
  "prompt_text": str       # Optional prompt for task generation
}
```

### ScheduleResponse
```python
{
  "tasks": List[List[float]]  # List of task priority arrays
}
```

### EvaluateRequest
```python
{
  "prompt_text": str,      # Optional evaluation prompt
  "num_episodes": int      # Number of episodes to evaluate
}
```

### EvaluateResponse
```python
{
  "average_reward": float   # Average reward across episodes
}
```

### ChartDataResponse
```python
{
  "chart_data": str        # JSON string for Chart.js
}
```

---

## Rate Limiting
Currently no rate limiting is implemented. Consider adding rate limiting for production use.

## Versioning
- Current version: 1.0.0
- API versioning can be added using URL prefixes (e.g., `/v1/prompt/handle`)

## SDKs and Libraries
- Python: Use `requests` or `httpx` for API calls
- JavaScript: Use `fetch` or `axios`
- Chart.js: Required for visualization endpoints

## Support
For issues or questions, check the application logs or contact the development team.