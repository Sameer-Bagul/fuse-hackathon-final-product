# API Documentation

## Overview

### Architecture and Proxy Design

This API implements a **Proxy Design Pattern** where the Node.js Express server acts as an intermediary between the client application and the Python ML backend services. The proxy pattern provides several architectural benefits:

#### Key Architectural Components

1. **Express Server (Proxy Layer)**
   - Built with Node.js and Express.js
   - Handles HTTP requests/responses, authentication, and validation
   - Implements comprehensive middleware chain for security and monitoring
   - Forwards validated requests to the ML backend

2. **ML Backend (Service Layer)**
   - Python-based machine learning services
   - Runs on port 8081 (localhost:8081)
   - Handles computationally intensive ML operations
   - Provides RESTful endpoints for ML functionality

3. **Client Application (Frontend)**
   - React/TypeScript application
   - Communicates exclusively with the Express proxy
   - Receives standardized JSON responses

#### Proxy Design Pattern Benefits

- **Abstraction**: Clients don't need to know about ML backend implementation details
- **Centralized Control**: Single point for authentication, validation, and error handling
- **Resilience**: Circuit breaker patterns and timeout management
- **Monitoring**: Comprehensive logging and metrics collection
- **Security**: Input validation and sanitization before reaching ML services
- **Scalability**: Easy to add load balancing or multiple ML backend instances

#### Data Flow Architecture

```
Client Request → Express Server → Validation → Authentication → Rate Limiting → ML Backend → Response Processing → Client
```

#### MVC Integration

The API follows the Model-View-Controller pattern:

- **Models**: Joi validation schemas and data structures (`server/models/`)
- **Views**: Response formatting utilities (`server/views/`)
- **Controllers**: Business logic and proxy handling (`server/controllers/`)

#### Middleware Chain

All ML endpoints use a comprehensive middleware chain:
1. `requestLogger` - Logs incoming requests
2. `responseLogger` - Logs responses with performance metrics
3. `securityLogger` - Monitors security events
4. `mlRateLimiter` - Rate limiting (50 requests/hour for ML operations)
5. `userAuth` - JWT authentication
6. `validateRequest` - Input validation using Joi schemas
7. `mlController` - Business logic and ML backend proxying
8. `mlErrorHandler` - Centralized error handling

#### Dependencies

- **Runtime**: Node.js, Python 3.x
- **Web Framework**: Express.js
- **Validation**: Joi
- **Authentication**: JWT (jsonwebtoken)
- **HTTP Client**: Axios
- **Database**: MongoDB (for user management)
- **ML Backend**: Custom Python services

---

## API Endpoints

### Base URL
```
http://localhost:3000/api/ml
```

### Authentication
All endpoints require JWT authentication via cookie-based tokens. Include the `token` cookie in requests.

### Response Format
All responses follow a consistent structure:
```json
{
  "success": boolean,
  "data": object | null,
  "message": string (only on errors)
}
```

---

## 1. Prompt Handling Endpoints

### 1.1 Handle Prompt Processing
**Endpoint:** `POST /api/ml/prompt/handle`  
**Purpose:** Process user prompts through the LLM learning system and collect learning data

#### Request Structure
```json
{
  "prompt_text": "string (required, 1-10000 chars)",
  "metadata": {
    "user_id": "string (optional)",
    "session_id": "string (optional)"
  }
}
```

#### Response Structure
```json
{
  "success": true,
  "data": {
    "response": "string",
    "action": number,
    "metrics": {
      "success_rate": number,
      "pattern_frequency": object,
      "total_interactions": number
    }
  }
}
```

#### Data Flow
1. Client sends prompt → Express validation → JWT auth check
2. Rate limiting applied → Forward to ML backend (`POST /prompt/handle`)
3. ML backend processes prompt → Returns learning data
4. Express formats response → Returns to client

#### Error Handling
- **400**: Invalid prompt_text (empty, too long, not string)
- **401**: Authentication failed
- **429**: Rate limit exceeded (50/hour)
- **504**: ML backend timeout (30s)
- **500**: Internal server error

#### Middleware
- `userAuth`: JWT token validation
- `validateRequest`: Joi schema validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Model**: `promptRequestSchema` validation
- **Controller**: `handlePrompt` function with axios proxy
- **View**: Standardized JSON response formatting

#### Dependencies
- ML Backend: `/prompt/handle` endpoint
- Auth: JWT verification
- Validation: Joi prompt schema

#### Example Request
```bash
curl -X POST http://localhost:3000/api/ml/prompt/handle \
  -H "Content-Type: application/json" \
  -H "Cookie: token=your_jwt_token" \
  -d '{
    "prompt_text": "Explain machine learning",
    "metadata": {
      "user_id": "user123",
      "session_id": "session456"
    }
  }'
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "response": "Machine learning is a subset of AI...",
    "action": 1,
    "metrics": {
      "success_rate": 0.85,
      "pattern_frequency": {"explain": 5, "learning": 3},
      "total_interactions": 150
    }
  }
}
```

---

## 2. Metrics Endpoints

### 2.1 Get Learning Metrics
**Endpoint:** `GET /api/ml/metrics`  
**Purpose:** Retrieve current learning metrics and performance data from the ML system

#### Request Structure (Query Parameters)
```
filters[user_id]: string (optional)
filters[session_id]: string (optional)
filters[date_range][start]: ISO date string (optional)
filters[date_range][end]: ISO date string (optional)
```

#### Response Structure
```json
{
  "success": true,
  "data": {
    "success_rates": [0.8, 0.85, 0.9],
    "patterns": {
      "pattern1": 10,
      "pattern2": 5
    },
    "total_interactions": 150,
    "average_reward": 0.82
  }
}
```

#### Data Flow
1. Client requests metrics → Query param validation → Auth check
2. Rate limiting → Forward to ML backend (`GET /prompt/metrics`)
3. ML backend aggregates metrics → Returns data
4. Express formats response → Returns to client

#### Error Handling
- **400**: Invalid query parameters
- **401**: Authentication failed
- **429**: Rate limit exceeded
- **504**: ML backend timeout
- **500**: Internal server error

#### Middleware
- `userAuth`: JWT token validation
- `validateRequest`: Metrics query schema validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Model**: `metricsRequestSchema` validation
- **Controller**: `getLearningMetrics` with axios proxy
- **View**: JSON response formatting

#### Dependencies
- ML Backend: `/prompt/metrics` endpoint
- Auth: JWT verification
- Validation: Joi metrics schema

#### Example Request
```bash
curl -X GET "http://localhost:3000/api/ml/metrics?filters[user_id]=user123&filters[date_range][start]=2024-01-01T00:00:00Z" \
  -H "Cookie: token=your_jwt_token"
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "success_rates": [0.75, 0.82, 0.88, 0.91],
    "patterns": {
      "question": 25,
      "explanation": 18,
      "code": 12
    },
    "total_interactions": 200,
    "average_reward": 0.84
  }
}
```

---

## 3. Evaluation Endpoints

### 3.1 Evaluate Model Performance
**Endpoint:** `POST /api/ml/evaluate`  
**Purpose:** Run evaluation episodes to assess LLM learning effectiveness and measure success rates

#### Request Structure
```json
{
  "llm_config": "string or object (required)",
  "task": "string or array (required)",
  "num_episodes": number (optional, default: 100, max: 10000),
  "settings": {
    "random_seed": number (optional),
    "timeout_seconds": number (optional, default: 300)
  }
}
```

#### Response Structure
```json
{
  "success": true,
  "data": {
    "average_reward": number,
    "total_episodes": number,
    "performance_metrics": {
      "min_reward": number,
      "max_reward": number,
      "reward_variance": number,
      "success_rate": number
    },
    "metadata": {
      "evaluation_duration_seconds": number,
      "timestamp": "ISO date string"
    }
  }
}
```

#### Data Flow
1. Client sends evaluation request → Schema validation → Auth check
2. Rate limiting → Forward to ML backend (`POST /evaluate`)
3. ML backend runs evaluation episodes → Returns results
4. Express formats response → Returns to client

#### Error Handling
- **400**: Invalid evaluation parameters
- **401**: Authentication failed
- **429**: Rate limit exceeded
- **504**: Evaluation timeout (30s)
- **500**: Internal server error

#### Middleware
- `userAuth`: JWT token validation
- `validateRequest`: Evaluation schema validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Model**: `evaluationRequestSchema` validation
- **Controller**: `evaluateModel` with axios proxy
- **View**: JSON response formatting

#### Dependencies
- ML Backend: `/evaluate` endpoint
- Auth: JWT verification
- Validation: Joi evaluation schema

#### Example Request
```bash
curl -X POST http://localhost:3000/api/ml/evaluate \
  -H "Content-Type: application/json" \
  -H "Cookie: token=your_jwt_token" \
  -d '{
    "llm_config": "gpt-3.5-turbo",
    "task": "math_problem_solving",
    "num_episodes": 50,
    "settings": {
      "random_seed": 42
    }
  }'
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "average_reward": 0.78,
    "total_episodes": 50,
    "performance_metrics": {
      "min_reward": 0.45,
      "max_reward": 0.95,
      "reward_variance": 0.12,
      "success_rate": 0.76
    },
    "metadata": {
      "evaluation_duration_seconds": 45.2,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }
}
```

---

## 4. Scheduling Endpoints

### 4.1 Schedule Learning Tasks
**Endpoint:** `POST /api/ml/schedule`  
**Purpose:** Generate and schedule learning tasks for the LLM system to create structured curricula

#### Request Structure
```json
{
  "num_tasks": number (required, 1-1000),
  "prompt": {
    "text": "string (optional, max 10000 chars)",
    "metadata": object (optional)
  },
  "constraints": {
    "max_difficulty": number (optional, 0-1),
    "required_skills": array (optional),
    "time_limit_minutes": number (optional, max 480)
  }
}
```

#### Response Structure
```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "id": "string",
        "description": "string",
        "type": "prompt_response",
        "difficulty": number,
        "priority_score": number,
        "estimated_time": number,
        "metadata": object
      }
    ],
    "metadata": {
      "total_tasks_requested": number,
      "total_tasks_generated": number,
      "generation_time_seconds": number
    },
    "statistics": {
      "average_difficulty": number,
      "task_types": object
    }
  }
}
```

#### Data Flow
1. Client sends scheduling request → Schema validation → Auth check
2. Rate limiting → Forward to ML backend (`POST /schedule`)
3. ML backend generates task curriculum → Returns tasks
4. Express formats response → Returns to client

#### Error Handling
- **400**: Invalid scheduling parameters
- **401**: Authentication failed
- **429**: Rate limit exceeded
- **504**: Task generation timeout (30s)
- **500**: Internal server error

#### Middleware
- `userAuth`: JWT token validation
- `validateRequest`: Scheduling schema validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Model**: `schedulingRequestSchema` validation
- **Controller**: `scheduleTasks` with axios proxy
- **View**: JSON response formatting

#### Dependencies
- ML Backend: `/schedule` endpoint
- Auth: JWT verification
- Validation: Joi scheduling schema

#### Example Request
```bash
curl -X POST http://localhost:3000/api/ml/schedule \
  -H "Content-Type: application/json" \
  -H "Cookie: token=your_jwt_token" \
  -d '{
    "num_tasks": 10,
    "prompt": {
      "text": "Focus on mathematics and logic problems"
    },
    "constraints": {
      "max_difficulty": 0.8,
      "required_skills": ["algebra", "logic"]
    }
  }'
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "id": "task_001",
        "description": "Solve quadratic equations",
        "type": "prompt_response",
        "difficulty": 0.6,
        "priority_score": 0.85,
        "estimated_time": 15,
        "metadata": {
          "domain": "mathematics",
          "category": "algebra"
        }
      }
    ],
    "metadata": {
      "total_tasks_requested": 10,
      "total_tasks_generated": 10,
      "generation_time_seconds": 2.3
    },
    "statistics": {
      "average_difficulty": 0.65,
      "task_types": {
        "prompt_response": 8,
        "evaluation": 2
      }
    }
  }
}
```

---

## 5. Visualization Endpoints

### 5.1 Get Chart.js Visualization Data
**Endpoint:** `GET /api/ml/visualize/chartjs`  
**Purpose:** Retrieve Chart.js compatible data for interactive frontend visualizations

#### Request Structure (Query Parameters)
```
type: "chartjs_data" (required)
rewards: array of numbers (required)
config: visualization configuration object (optional)
```

#### Response Structure
```json
{
  "success": true,
  "data": {
    "learning_curve": {
      "labels": ["Episode 1", "Episode 2", ...],
      "datasets": [
        {
          "label": "Rewards",
          "data": [0.5, 0.7, 0.8, ...],
          "borderColor": "rgb(75, 192, 192)",
          "tension": 0.1
        }
      ]
    },
    "success_rate": {
      "labels": ["Episode 1-10", "Episode 11-20", ...],
      "datasets": [...]
    }
  }
}
```

#### Data Flow
1. Client requests chart data → Query validation → Auth check
2. Rate limiting → Forward to ML backend (`GET /visualize/chartjs`)
3. ML backend generates Chart.js data → Returns visualization data
4. Express formats response → Returns to client

#### Error Handling
- **400**: Invalid visualization parameters
- **401**: Authentication failed
- **429**: Rate limit exceeded
- **504**: Chart generation timeout (30s)
- **500**: Internal server error

#### Middleware
- `userAuth`: JWT token validation
- `validateRequest`: Visualization schema validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Model**: `visualizationRequestSchema` validation
- **Controller**: `getChartData` with axios proxy
- **View**: JSON response formatting

#### Dependencies
- ML Backend: `/visualize/chartjs` endpoint
- Auth: JWT verification
- Validation: Joi visualization schema

#### Example Request
```bash
curl -X GET "http://localhost:3000/api/ml/visualize/chartjs?type=chartjs_data&rewards[]=0.5&rewards[]=0.7&rewards[]=0.8" \
  -H "Cookie: token=your_jwt_token"
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "learning_curve": {
      "labels": ["Episode 1", "Episode 2", "Episode 3"],
      "datasets": [
        {
          "label": "Learning Rewards",
          "data": [0.5, 0.7, 0.8],
          "borderColor": "rgb(75, 192, 192)",
          "backgroundColor": "rgba(75, 192, 192, 0.2)",
          "tension": 0.1
        }
      ]
    }
  }
}
```

### 5.2 Get Learning Progress Data
**Endpoint:** `GET /api/ml/visualize/learning_progress`  
**Purpose:** Retrieve comprehensive learning progress information for detailed analysis

#### Response Structure
```json
{
  "success": true,
  "data": {
    "progress_metrics": {
      "total_episodes": number,
      "current_success_rate": number,
      "improvement_rate": number
    },
    "timeline": [
      {
        "timestamp": "ISO date string",
        "episode": number,
        "reward": number,
        "success": boolean
      }
    ]
  }
}
```

#### Data Flow
1. Client requests progress data → Auth check
2. Rate limiting → Forward to ML backend (`GET /visualize/learning_progress`)
3. ML backend aggregates progress data → Returns comprehensive metrics
4. Express formats response → Returns to client

#### Error Handling
- **401**: Authentication failed
- **429**: Rate limit exceeded
- **504**: Data retrieval timeout (30s)
- **500**: Internal server error

#### Middleware
- `userAuth`: JWT token validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Controller**: `getLearningProgress` with axios proxy
- **View**: JSON response formatting

#### Dependencies
- ML Backend: `/visualize/learning_progress` endpoint
- Auth: JWT verification

#### Example Request
```bash
curl -X GET http://localhost:3000/api/ml/visualize/learning_progress \
  -H "Cookie: token=your_jwt_token"
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "progress_metrics": {
      "total_episodes": 150,
      "current_success_rate": 0.82,
      "improvement_rate": 0.15
    },
    "timeline": [
      {
        "timestamp": "2024-01-15T09:00:00Z",
        "episode": 145,
        "reward": 0.85,
        "success": true
      }
    ]
  }
}
```

### 5.3 Generate Learning Report
**Endpoint:** `POST /api/ml/visualize/generate_report`  
**Purpose:** Create comprehensive learning reports with visualizations and insights

#### Response Structure
```json
{
  "success": true,
  "data": {
    "summary": {
      "total_episodes": number,
      "average_reward": number,
      "success_rate": number
    },
    "charts": {
      "learning_curve": "base64_encoded_image_or_json",
      "success_rate": "base64_encoded_image_or_json"
    },
    "insights": [
      "Learning performance has improved by 15% over the last 50 episodes",
      "Success rate is highest for pattern recognition tasks"
    ]
  }
}
```

#### Data Flow
1. Client requests report generation → Auth check
2. Rate limiting → Forward to ML backend (`POST /visualize/generate_report`)
3. ML backend generates comprehensive report → Returns analysis data
4. Express formats response → Returns to client

#### Error Handling
- **401**: Authentication failed
- **429**: Rate limit exceeded
- **504**: Report generation timeout (30s)
- **500**: Internal server error

#### Middleware
- `userAuth`: JWT token validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Controller**: `generateReport` with axios proxy
- **View**: JSON response formatting

#### Dependencies
- ML Backend: `/visualize/generate_report` endpoint
- Auth: JWT verification

#### Example Request
```bash
curl -X POST http://localhost:3000/api/ml/visualize/generate_report \
  -H "Cookie: token=your_jwt_token"
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "summary": {
      "total_episodes": 200,
      "average_reward": 0.78,
      "success_rate": 0.75
    },
    "charts": {
      "learning_curve": "iVBORw0KGgoAAAANSUhEUgAA...",
      "success_rate": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    "insights": [
      "Overall learning trend shows consistent improvement",
      "Peak performance achieved at episode 175",
      "Consider increasing task difficulty for continued growth"
    ]
  }
}
```

---

## 6. Health Check Endpoints

### 6.1 Check ML Backend Health
**Endpoint:** `GET /api/ml/health`  
**Purpose:** Monitor ML backend service availability and status

#### Response Structure
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "string",
    "uptime": number,
    "last_check": "ISO date string"
  }
}
```

#### Data Flow
1. Client requests health check → Auth check
2. Rate limiting → Forward to ML backend (`GET /health`)
3. ML backend returns health status → Express formats response
4. Returns to client

#### Error Handling
- **401**: Authentication failed
- **429**: Rate limit exceeded
- **504**: Health check timeout (30s)
- **500**: ML backend unavailable

#### Middleware
- `userAuth`: JWT token validation
- `mlRateLimiter`: 50 requests/hour limit

#### MVC Integration
- **Controller**: `checkHealth` with axios proxy
- **View**: JSON response formatting

#### Dependencies
- ML Backend: `/health` endpoint
- Auth: JWT verification

#### Example Request
```bash
curl -X GET http://localhost:3000/api/ml/health \
  -H "Cookie: token=your_jwt_token"
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.2.3",
    "uptime": 3600,
    "last_check": "2024-01-15T10:30:00Z"
  }
}
```

---

## Error Response Format

All error responses follow a consistent structure:

```json
{
  "success": false,
  "message": "Human-readable error description",
  "errors": [
    {
      "field": "field_name",
      "message": "Specific validation error",
      "value": "provided_value"
    }
  ]
}
```

## Rate Limiting

- **ML Operations**: 50 requests per hour per user
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- **429 Status**: Returned when limits exceeded

## Security Features

- JWT-based authentication
- Input validation and sanitization
- Rate limiting protection
- CORS configuration
- Request/response logging
- SQL injection prevention
- XSS protection

## Performance Considerations

- 30-second timeout for ML operations
- Request/response logging for monitoring
- Circuit breaker pattern support
- Connection pooling with axios
- Memory-efficient validation

## Monitoring and Logging

- Request/response logging
- Security event monitoring
- Performance metrics collection
- Error tracking and reporting
- Rate limit violation logging

---

*This documentation is auto-generated from the codebase and reflects the current API implementation as of the last code analysis.*