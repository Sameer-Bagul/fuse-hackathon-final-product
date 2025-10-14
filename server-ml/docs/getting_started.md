# Getting Started with LLM Learning System API

## Prerequisites

Before running the LLM Learning System API, ensure you have:

- Python 3.8 or higher
- pip package manager
- Basic understanding of REST APIs and MVC architecture

## Installation

### 1. Clone or navigate to the project directory
```bash
cd path/to/your/project/Ml_backend
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

This will install:
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation and serialization
- **Matplotlib**: Data visualization
- **NumPy**: Numerical computing
- **Requests**: HTTP client for external API calls

## Configuration

### Environment Variables (Optional)
You can customize the application behavior using environment variables:

```bash
# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8081"
export API_DEBUG="true"

# LLM Configuration
export LLM_NUM_ACTIONS="10"
export LLM_ALPHA="0.1"
export LLM_EPSILON="0.1"

# Logging
export LOG_LEVEL="INFO"
```

### Configuration File (Optional)
Create a `config.json` file in the project root:

```json
{
  "llm": {
    "num_actions": 10,
    "alpha": 0.1,
    "epsilon": 0.1
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8081,
    "debug": true
  }
}
```

## Running the Application

### Development Mode (with auto-reload)
```bash
python app.py
```

### Production Mode (recommended for production)
```bash
# Using Uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8081 --workers 4

# Or using Gunicorn (more production-ready)
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8081
```

### Using Docker (optional)
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8081

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t llm-api .
docker run -p 8081:8081 llm-api
```

## Testing the API

### Health Check
```bash
curl http://localhost:8081/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "version": "1.0.0",
  "environment": "development"
}
```

### Interactive API Documentation
Visit `http://localhost:8081/docs` in your browser for:
- Interactive API documentation (Swagger UI)
- Try out endpoints directly
- View request/response schemas

Alternative: `http://localhost:8081/redoc` for ReDoc documentation.

### Basic API Testing

#### 1. Handle a Prompt
```bash
curl -X POST "http://localhost:8081/prompt/handle" \
  -H "Content-Type: application/json" \
  -d '{"prompt_text": "Hello, how are you?"}'
```

#### 2. Get Learning Metrics
```bash
curl http://localhost:8081/prompt/metrics
```

#### 3. Schedule Tasks
```bash
curl -X POST "http://localhost:8081/schedule" \
  -H "Content-Type: application/json" \
  -d '{"num_tasks": 3, "prompt_text": "Learn about AI"}'
```

#### 4. Run Evaluation
```bash
curl -X POST "http://localhost:8081/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"num_episodes": 50}'
```

#### 5. Get Visualization Data
```bash
curl http://localhost:8081/visualize/chartjs
```

## Understanding the MVC Architecture

### Model Layer (`models/`)
Contains the business logic and data structures:
- `LLM`: The learning agent with Q-learning
- `History`: Tracks interactions and learning progress
- `Prompt`: Represents user input
- `TaskGenerator`: Creates learning tasks

### Controller Layer (`controllers/`)
Handles API requests and orchestrates operations:
- `PromptController`: Manages prompt processing
- `Evaluator`: Runs performance evaluations
- `Scheduler`: Generates and prioritizes tasks

### Service Layer (`services/`)
Additional business logic and utilities:
- `VisualizationService`: Handles data visualization and reporting

### View Layer (API Responses)
The API endpoints serve as the "View" layer, formatting data for consumption.

## Development Workflow

### 1. Make Changes
Edit the code files in your preferred IDE.

### 2. Test Locally
```bash
# Run with auto-reload
python app.py

# Test endpoints
curl http://localhost:8081/health
```

### 3. Check Logs
Logs are written to `logs/llm_learning.log` and console.

### 4. Run Tests (if available)
```bash
pytest
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8081
lsof -i :8081

# Kill the process
kill -9 <PID>

# Or use a different port
API_PORT=8082 python app.py
```

#### Import Errors
```bash
# Ensure you're in the correct directory
cd Ml_backend

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Configuration Issues
```bash
# Check environment variables
env | grep -E "(API|LLM|LOG)_"

# Validate config file
python -c "import json; print(json.load(open('config.json')))"
```

### Performance Tuning

#### For Development
- Use `reload=True` in Uvicorn for auto-reload
- Set `LOG_LEVEL=DEBUG` for detailed logging

#### For Production
- Use multiple workers: `uvicorn app:app --workers 4`
- Set appropriate log levels
- Configure reverse proxy (nginx)
- Enable gzip compression

## Next Steps

1. **Explore the Code**: Read through the source files to understand the implementation
2. **Add Features**: Implement new endpoints or enhance existing ones
3. **Add Tests**: Write unit and integration tests
4. **Deploy**: Set up production deployment with proper monitoring
5. **Scale**: Add database persistence, caching, and horizontal scaling

## Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MVC Architecture Pattern](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller)
- [Reinforcement Learning](https://www.coursera.org/learn/reinforcement-learning)
- [REST API Design](https://restfulapi.net/)

## Support

- Check the logs in `logs/` directory
- Review the API documentation at `/docs`
- Examine the code comments for implementation details