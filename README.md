# 📚 Fuse Hackathon - Autonomous Learning System# MERN Authentication System with Advanced Features



## 🎯 What Is This?A robust authentication system built with the MERN stack (MongoDB, Express.js, React.js, Node.js) featuring advanced security measures and user verification methods.



An **autonomous AI learning system** that uses reinforcement learning (PPO), meta-learning, and curriculum generation to create a self-improving AI agent. The system learns from user interactions and generates its own training prompts automatically.## 🚀 Features



### Key Features:- **User Authentication**

- 🚀 **User-Driven Start**: System waits for your initial prompt  - Email & Password based authentication

- 🤖 **Autonomous Learning**: LLM generates all subsequent prompts automatically  - JWT (JSON Web Token) based authorization

- 📊 **Real-Time Dashboard**: Monitor learning progress, metrics, and curriculum  - Password hashing and security

- 🎓 **Curriculum Generation**: Creates structured learning paths from your prompts  - Password reset functionality

- 🧠 **Meta-Learning**: Adapts learning strategies based on performance  

- 💾 **Persistent State**: Saves progress to MongoDB- **Advanced Security**

  - OTP (One-Time Password) verification

---  - Email verification

  - Session management

## 🚀 Quick Start  - Protected routes

  

### Prerequisites- **Email Integration**

- Python 3.10+  - SMTP email service integration

- Node.js 16+  - Email notifications for:

- MongoDB Atlas account (free tier works)    - Account verification

    - Password reset

### 1. Backend Setup    - Security alerts

    - OTP delivery

```bash    

cd server-ml- **Frontend Features**

  - React.js with Context API for state management

# Create virtual environment  - Responsive design

python3 -m venv venv  - Form validation

source venv/bin/activate  # On Windows: venv\Scripts\activate  - Protected route components

  - User dashboard

# Install dependencies  

pip install -r requirements.txt- **Backend Features**

  - Express.js REST API

# Configure environment  - MongoDB database integration

cp .env.example .env  - Middleware for authentication

# Edit .env with your MongoDB connection string  - Rate limiting

  - Input sanitization

# Start server

python app.py## 📋 Prerequisites

```

Before you begin, ensure you have the following installed:

Server runs at: `http://localhost:8082`- Node.js (v14 or higher)

- MongoDB

### 2. Frontend Setup- npm or yarn

- Git

```bash

cd frontend## 🛠️ Installation



# Install dependencies1. Clone the repository:

npm install   ```bash

   git clone https://github.com/Sameer-Bagul/best-mern-auth.git

# Start dev server   cd best-mern-auth

npm run dev   ```

```

2. Install dependencies for both frontend and backend:

Frontend runs at: `http://localhost:5173`   ```bash

   # Install backend dependencies

### 3. Test the System   cd backend

   npm install

1. Open browser to `http://localhost:5173`

2. Go to **Learning Control** panel   # Install frontend dependencies

3. Submit an initial prompt (e.g., "Teach me Python programming")   cd ../frontend

4. Watch the autonomous learning begin! 🎉   npm install

   ```

---

3. Create a `.env` file in the backend directory with the following variables:

## 📖 Documentation   ```env

   PORT=5000

**👉 See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for complete documentation**   MONGODB_URI=your_mongodb_uri

   JWT_SECRET=your_jwt_secret

### Quick Links:   SMTP_HOST=your_smtp_host

- **[Initial Prompt Examples](INITIAL_PROMPT_EXAMPLES.md)** - Great prompts to try   SMTP_PORT=your_smtp_port

- **[Dashboard Guide](DASHBOARD_GUIDE.md)** - How to use the UI   SMTP_USER=your_smtp_email

- **[API Documentation](API_DOCUMENTATION.md)** - API reference   SMTP_PASS=your_smtp_password

- **[Troubleshooting](docs/troubleshooting/COMMON_ISSUES.md)** - Fix common issues   CLIENT_URL=http://localhost:3000

   ```

---

4. Create a `.env` file in the frontend directory:

## 🎯 How It Works   ```env

   REACT_APP_API_URL=http://localhost:5000/api

### 1. User Submits Initial Prompt   ```

```

You: "Teach me Python programming, starting with basics, then OOP"## 🚀 Running the Application

```

1. Start the backend server:

### 2. System Generates Curriculum   ```bash

```   cd backend

✅ Curriculum Created:   npm run dev

   - Task 1: Python Basics (Variables, Data Types)   ```

   - Task 2: Functions and Control Flow

   - Task 3: Object-Oriented Programming2. Start the frontend development server:

```   ```bash

   cd frontend

### 3. Autonomous Learning Loop Starts   npm start

```   ```

🔄 Iteration 1: Processing Task 1...

🔄 Iteration 2: Processing Task 2...The application will be available at:

🔄 Iteration 3: Processing Task 3...- Frontend: `http://localhost:3000`

🤖 Iteration 4: LLM generates new prompt automatically- Backend API: `http://localhost:5000`

🤖 Iteration 5: LLM generates another prompt

... continues indefinitely## 📚 API Documentation

```

### Authentication Endpoints

### 4. You Monitor Progress

- View real-time metrics on dashboard- `POST /api/auth/register` - Register a new user

- See learning history (user vs AI prompts)- `POST /api/auth/login` - Login user

- Track curriculum progress- `POST /api/auth/verify-email` - Verify email address

- Monitor success rates- `POST /api/auth/forgot-password` - Request password reset

- `POST /api/auth/reset-password` - Reset password

---- `POST /api/auth/verify-otp` - Verify OTP

- `GET /api/auth/profile` - Get user profile (protected)

## 🏗️ Project Structure

## 🔒 Security Features

```

fuse-hackathon/- Password hashing using bcrypt

├── server-ml/              # Backend Python server- JWT token authentication

│   ├── app.py             # Main FastAPI application- Email verification

│   ├── models/            # ML models (PPO, Q-learning)- OTP verification

│   ├── services/          # Business logic services- Rate limiting on API endpoints

│   ├── controllers/       # Request handlers- Input validation and sanitization

│   └── utils/             # Utilities and config- Protected routes on both frontend and backend

├── frontend/              # React TypeScript UI- HTTP-only cookies for token storage

│   ├── src/

│   │   ├── components/    # React components## 🛠️ Built With

│   │   ├── pages/        # Page components

│   │   └── lib/          # API client and utilities- **Frontend**

│   └── public/           # Static assets  - React.js

└── docs/                 # Documentation  - Context API

    ├── guides/           # User guides  - Axios

    ├── troubleshooting/  # Problem solutions  - React Router

    └── reference/        # Technical docs  - Material-UI/Tailwind CSS

```

- **Backend**

---  - Node.js

  - Express.js

## 🎨 Tech Stack  - MongoDB

  - Mongoose

### Backend  - JWT

- **FastAPI** - High-performance async API  - Nodemailer

- **PyTorch** - Deep learning framework (PPO agent)  - bcrypt

- **MongoDB** - Persistent data storage

- **NumPy** - Numerical computations## 🤝 Contributing



### Frontend1. Fork the repository

- **React** - UI framework2. Create your feature branch (`git checkout -b feature/AmazingFeature`)

- **TypeScript** - Type safety3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

- **TailwindCSS** - Styling4. Push to the branch (`git push origin feature/AmazingFeature`)

- **Radix UI** - Component library5. Open a Pull Request

- **Vite** - Build tool

## 📝 License

---

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Dashboard Components

## 📧 Contact

1. **Learning Control** - Start/stop learning, submit prompts

2. **Metrics Overview** - Success rate, iterations, rewardsYour Name - sameerbagul2004@gmail.com

3. **Learning History** - View all interactions (user + AI)Project Link: https://github.com/Sameer-Bagul/best-mern-auth.git

4. **Curriculum Progress** - Track skill mastery

5. **Hallucination Monitor** - Detect unreliable responses## 🙏 Acknowledgments

6. **Reward System** - View reward metrics

7. **Analytics** - Performance trends and insights- [Node.js](https://nodejs.org/)

- [React.js](https://reactjs.org/)

---- [Express.js](https://expressjs.com/)

- [MongoDB](https://www.mongodb.com/)

## 🎉 Success Metrics- [Nodemailer](https://nodemailer.com/) 

After starting the system:

✅ **Backend**: API running at port 8082  
✅ **Frontend**: UI accessible at port 5173  
✅ **Database**: Connected to MongoDB Atlas  
✅ **Learning Loop**: Green "Active" status  
✅ **Iterations**: Counter incrementing  
✅ **History**: AI-generated prompts appearing  

---

## 🐛 Common Issues

### Issue: "Cannot connect to MongoDB"
**Solution**: Check your `DATABASE_URL` in `.env` file

### Issue: "Port 8082 already in use"
**Solution**: Kill the process or change port in `.env`

### Issue: "Tensor size mismatch errors"
**Solution**: Restart server (fixes applied)

### Issue: "Loop not starting"
**Solution**: Make sure you submitted an initial prompt via UI

**More solutions**: [docs/troubleshooting/COMMON_ISSUES.md](docs/troubleshooting/COMMON_ISSUES.md)

---

## 📝 Environment Variables

Key variables in `server-ml/.env`:

```bash
# MongoDB Atlas
DATABASE_URL=mongodb+srv://user:pass@cluster.mongodb.net/

# API Configuration
API_HOST=0.0.0.0
API_PORT=8082

# External LLM (Optional)
EXTERNAL_LLM_ENABLED=false
OPENAI_API_KEY=your_key_here
```

---

## 📄 License

Hackathon Project - Feel free to use and modify!

---

**Built with ❤️ for Fuse Hackathon**

🚀 **[Get Started Now →](DOCUMENTATION_INDEX.md)**
