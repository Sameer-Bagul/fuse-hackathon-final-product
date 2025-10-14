# LLM Learning System FastAPI

This FastAPI application provides RESTful endpoints for the LLM Learning System, maintaining the MVC architecture.

## Installation

```bash
cd Ml_backend
pip install -r requirements.txt
```

## Running the API

```bash
python app.py
```

The API will be available at `http://localhost:8081`

## API Endpoints

### Prompt Handling

#### POST `/prompt/handle`
Handle a user prompt and get LLM response.

**Request Body:**
```json
{
  "prompt_text": "Your prompt here"
}
```

**Response:**
```json
{
  "response": "LLM response text",
  "action": 0
}
```

#### GET `/prompt/metrics`
Get learning metrics from prompt interactions.

**Response:**
```json
{
  "success_rate": 1.0,
  "pattern_frequency": {
    "word1": 5,
    "word2": 3
  },
  "total_interactions": 10
}
```

### Task Scheduling

#### POST `/schedule`
Generate and prioritize tasks based on a prompt.

**Request Body:**
```json
{
  "num_tasks": 3,
  "prompt_text": "Optional prompt for task generation"
}
```

**Response:**
```json
{
  "tasks": [
    [0.5, 0.7, 0.3, ...],
    [0.6, 0.4, 0.8, ...]
  ]
}
```

### Evaluation

#### POST `/evaluate`
Evaluate the LLM performance on tasks.

**Request Body:**
```json
{
  "prompt_text": "Optional prompt for evaluation",
  "num_episodes": 100
}
```

**Response:**
```json
{
  "average_reward": 0.65
}
```

### Visualization

#### GET `/visualize/chartjs`
Get Chart.js compatible data for learning progress visualization.

**Response:**
```json
{
  "chart_data": "{\"learning_curve\": {...}}"
}
```

#### GET `/visualize/learning_progress`
Get learning progress data.

**Response:**
```json
{
  "success_rates": [],
  "patterns": {"word": 1},
  "total_interactions": 5
}
```

## Architecture

The API maintains the MVC architecture:

- **Models**: Data and business logic (LLM, Prompt, History, TaskGenerator)
- **Controllers**: Handle API requests and orchestrate operations
- **Views**: Provide data for visualization (JSON responses for charts)

## Error Handling

All endpoints include proper error handling with HTTP status codes and descriptive error messages.

## Testing

Use the provided test scripts or curl commands to test the endpoints:

```bash
# Test prompt handling
curl -X POST "http://localhost:8081/prompt/handle" \
  -H "Content-Type: application/json" \
  -d '{"prompt_text": "Hello world"}'

# Test metrics
curl -X GET "http://localhost:8081/prompt/metrics"