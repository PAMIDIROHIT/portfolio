/**
 * PAMIDI ROHIT — Deep Research Knowledge Base
 * Built from actual codebase analysis of every GitHub repository
 * Each project entry reflects real code, architecture, and implementation details
 */

const KNOWLEDGE_BASE = [

  /* ═══════════════════════════════════════════════════════
     PERSONAL & CONTACT
  ═══════════════════════════════════════════════════════ */
  {
    id: "personal-intro",
    category: "personal",
    title: "Who is Pamidi Rohit",
    content: "Pamidi Rohit is a passionate B.Tech student specializing in Artificial Intelligence and Data Science at IIIT Sri City (Indian Institute of Information Technology, Sri City). He is a full-stack developer and ML engineer with deep expertise in building production-grade web applications, AI-powered systems, and machine learning pipelines. Rohit has 22+ GitHub repositories covering everything from microservices food delivery platforms with AI chatbots, to movie recommendation engines using content-based filtering, to enterprise NestJS REST APIs, to heart disease prediction with 6 competing ML models. His work spans React/Next.js, Node.js/NestJS/Flask backends, MongoDB/PostgreSQL databases, Docker deployments, and advanced ML with scikit-learn, XGBoost, and PyTorch. He solved 600+ LeetCode problems, built a Lunar Hazard Detection System using physics-informed neural networks for ISRO's Bharatiya Antariksh Hackathon 2025 (Team Lunar Pioneers, PS-11), and architected NeuroCore, a multi-agent AI smart city platform with 6 specialized agents (Team KernelCrew). His expertise extends to LangGraph for agentic AI workflows and GraphQL for flexible API design.",
    keywords: ["who", "about", "rohit", "pamidi", "introduce", "yourself", "tell me", "student", "developer", "engineer", "profile", "bio", "overview"]
  },
  {
    id: "contact-info",
    category: "contact",
    title: "Contact Information — Pamidi Rohit",
    content: "Pamidi Rohit contact details: Email: rohithtnsp@gmail.com | Phone and WhatsApp: +91 9398026237 | LinkedIn: https://www.linkedin.com/in/rohit-pamidi-4147771ba/ | GitHub: https://github.com/PAMIDIROHIT | LeetCode: https://leetcode.com/u/rohithtnsp/. He is actively seeking internships and full-time positions in software engineering, ML engineering, and full-stack development. Available for freelance projects and open-source collaboration.",
    keywords: ["contact", "email", "phone", "whatsapp", "linkedin", "github", "leetcode", "reach", "connect", "hire", "message", "call", "rohithtnsp"]
  },

  /* ═══════════════════════════════════════════════════════
     EDUCATION
  ═══════════════════════════════════════════════════════ */
  {
    id: "education-iiit",
    category: "education",
    title: "IIIT Sri City — B.Tech in AI and Data Science",
    content: "Pamidi Rohit is pursuing Bachelor of Technology in Artificial Intelligence and Data Science at Indian Institute of Information Technology IIIT Sri City, Andhra Pradesh. Duration: August 2023 to May 2027 Expected. Current CGPA: 8.23 out of 10.0. IIIT Sri City is a premier institute of national importance under the Ministry of Education, Government of India. The program focuses on machine learning, deep learning, data science, algorithms, probability and statistics, computer vision, NLP, software engineering, and cloud computing.",
    keywords: ["education", "college", "university", "iiit", "sri city", "btech", "b.tech", "artificial intelligence", "data science", "cgpa", "gpa", "degree", "graduation", "institute", "andhra pradesh"]
  },
  {
    id: "education-intermediate",
    category: "education",
    title: "Sri Chaitanya Junior College — Intermediate 93.7%",
    content: "Rohit completed Intermediate education at Sri Chaitanya Junior College with MPC Mathematics Physics Chemistry stream from June 2021 to June 2023. Score: 93.7% demonstrating strong mathematical and analytical aptitude that forms the foundation of his programming and ML skills.",
    keywords: ["intermediate", "school", "12th", "11th", "mpc", "sri chaitanya", "percentage", "93.7", "marks", "secondary education"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: TOMATO FOOD DELIVERY
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-tomato-food",
    category: "project",
    title: "Tomato — Full-Stack Food Delivery Platform with Gemini AI Chatbot",
    content: "Tomato is a production-grade full-stack food delivery platform with 3 independently deployable panels: Admin Panel React plus Vite, Customer Frontend React plus Vite, and a Node.js Express Backend. Architecture: The backend server.js creates an HTTP server with Socket.io for real-time WebSocket communication. Uses Express with Passport.js for OAuth authentication Google and Facebook strategies, JWT tokens for session management, and express-session with MongoStore for session persistence. CORS dynamically configured to allow multiple frontend origins on ports 5173 5174 5175 5176. Database MongoDB with Mongoose ODM. Models include: foodModel menu items with categories prices images, userModel authentication profile, orderModel order lifecycle with status tracking, conversationModel chatbot conversation history, deliveryPartnerModel rider management, notificationModel real-time alerts. Routes: food CRUD for menu, user authentication, cart management, order lifecycle, chatbot AI chat, auth OAuth, delivery rider tracking, notification push alerts, recommendation AI suggestions, images static uploads. AI Chatbot: Uses Google Gemini 1.5 Flash via google generative ai SDK. ChatbotService class fetches real-time menu from MongoDB with 5 minute cache, builds structured menu context grouped by category, passes it as context to Gemini AI for accurate menu queries order status checks and personalized food suggestions. Recommendation engine uses collaborative filtering on order history. WebSockets socket.js: Real-time order status updates pushed to customers, delivery partner location updates, admin order management. File uploads via multer to uploads directory. Redis caching reduces API response time from 800ms to 150ms achieving 81 percent improvement. Razorpay payment gateway integration. Email notifications via emailService. Docker plus CI/CD pipeline for deployment. GitHub: https://github.com/PAMIDIROHIT/Food-Delivery-MAIN1",
    keywords: ["tomato", "food delivery", "food", "ordering", "restaurant", "nodejs", "react", "mongodb", "redis", "docker", "microservices", "websocket", "socket.io", "jwt", "razorpay", "payment", "chatbot", "gemini", "passport", "oauth", "recommendation", "fullstack", "multer", "express", "mongoose", "food-delivery-main1"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: CINEMATCH AI
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-cinematch",
    category: "project",
    title: "CineMatch AI — Movie Recommendation System 10K+ Movies 95% Accuracy",
    content: "CineMatch AI is an intelligent movie recommendation platform with 2-tier architecture: Flask Python backend plus Next.js 14 TypeScript frontend. Backend app.py: Flask app with Flask-Caching Redis-backed 300s timeout for movie lists 600s for individual movies, Flask-Limiter rate limiting by IP, Flask-CORS. Two global pickle-loaded data structures: movies_df pandas DataFrame with 10000 plus movies and similarity numpy cosine similarity matrix. The generate_similarity.py script builds the similarity matrix using TF-IDF vectorization on movie features title genres keywords cast crew overview then computes cosine similarity. TMDB client utils/tmdb_client.py fetches real movie posters ratings release dates and cast from The Movie Database API using GENRE_IDS constant map. MongoDB database/mongodb.py stores user watchlists. API Endpoints: GET /api/movies paginated cached, GET /api/movie/id movie detail with TMDB data, POST /api/recommend content-based filtering finds movie by title in DataFrame retrieves top-N by cosine similarity scores, GET /api/search fuzzy title search, GET /api/genres all genre IDs, GET /api/random random movie discovery, GET /api/trending TMDB trending, POST DELETE GET /api/watchlist. Frontend Next.js 14: Pages include home with hero trending genre grid, discover browse all, search real-time search with debounced hook, movie detail dynamic route, watchlist. Components: hero-section trending-section genre-grid movie-card search-bar navbar footer. Custom React hooks: use-debounced-value for search debouncing, use-watchlist for watchlist state. Full SSR via Next.js App Router. Tailwind CSS styling. TypeScript throughout. Docker containerized. 95% recommendation accuracy on test set. GitHub: https://github.com/PAMIDIROHIT/cinematch-ai",
    keywords: ["cinematch", "movie", "recommendation", "film", "cinema", "tfidf", "cosine similarity", "flask", "nextjs", "next.js", "typescript", "tmdb", "content-based filtering", "ml", "nlp", "similarity matrix", "pickle", "pandas", "numpy", "redis", "mongodb", "watchlist", "10000 movies", "react", "tailwind", "ssr"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: NESTJS API
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-nestjs-api",
    category: "project",
    title: "NestJS Production REST API — JWT Auth PostgreSQL 51 Unit Tests",
    content: "A production-ready NestJS REST API showcasing enterprise-grade backend development. Tech stack: TypeScript NestJS framework PostgreSQL via Supabase cloud hosted JWT authentication Helmet for HTTP security headers Docker and docker-compose. Architecture: main.ts bootstraps NestFactory with global Helmet middleware CORS from ConfigService global ValidationPipe with whitelist true forbidNonWhitelisted true transform true enableImplicitConversion true and global HttpExceptionFilter for standardized error responses. Module structure: AppModule root, auth module JWT strategy login register guards, tasks module CRUD with full lifecycle OPEN IN_PROGRESS DONE status, users module user management, config module env configuration, common module filters interceptors decorators. Database TypeORM with Supabase PostgreSQL entity definitions migrations. Authentication flow: register hash password save user login validate credentials issue JWT protected routes via JwtAuthGuard. Task management CRUD: Create task get all with filtering by status and search get by ID update status delete. Testing: 51 comprehensive unit tests with Jest covering all service methods controllers guards and filters. Dockerfile plus docker-compose for containerized deployment. Full architectural documentation with UML diagrams in DOCUMENTATION.md. Port 3000 configurable via ConfigService. GitHub: https://github.com/PAMIDIROHIT/nestjs-api-project",
    keywords: ["nestjs", "nest", "rest api", "postgresql", "postgres", "supabase", "docker", "jwt", "authentication", "typescript", "testing", "unit tests", "51 tests", "jest", "backend api", "crud", "production", "helmet", "typeorm", "validation pipe", "enterprise", "tasks", "modules"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: HEART DISEASE ML
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-heart-disease",
    category: "project",
    title: "HeartDisease ML FullStack — 6 Models XGBoost 90%+ Accuracy AWS Deployed",
    content: "HeartDisease ML FullStack is an AI-powered medical prediction system with a sophisticated ML pipeline. Backend structure: Flask API with source organized into data_processing model_training prediction and services modules. ML Pipeline train.py trains and compares 6 models simultaneously: LogisticRegression max_iter 1000, DecisionTreeClassifier, RandomForestClassifier with 100 estimators, SVC with probability True, XGBClassifier, and MLPClassifier neural network max_iter 1000. Each model evaluated with 5-fold cross-validation cross_val_score reporting CV mean and standard deviation. GridSearchCV performs hyperparameter tuning on the winning model. Metrics: accuracy_score precision_score recall_score f1_score roc_auc_score confusion_matrix visualized with seaborn heatmap. Feature engineering on UCI Heart Disease Dataset with 13 clinical features: age sex chest pain type trestbps chol fbs restecg thalach exang oldpeak slope ca thal. RandomForest achieves 90 plus percent accuracy after tuning. Data pipeline handles missing values categorical encoding feature scaling with StandardScaler. React frontend with Chart.js health metric visualizations and prediction results display. Docker containerization. AWS EC2 deployment with 40 percent latency reduction through optimization. RESTful Flask API endpoints for prediction requests. Makefile for one-command builds. GitHub: https://github.com/PAMIDIROHIT/HeartDisease-ML-FullStack",
    keywords: ["heart disease", "health prediction", "medical", "healthcare", "prediction", "machine learning", "flask", "python", "react", "aws", "random forest", "xgboost", "svm", "logistic regression", "decision tree", "neural network", "uci dataset", "accuracy", "fullstack", "clinical", "cross validation", "gridsearchcv", "scikit-learn", "matplotlib", "seaborn", "docker", "ec2"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: WANDERLUST
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-wanderlust",
    category: "project",
    title: "WanderLust — Full-Stack Airbnb-Like Travel Platform with Stripe and AI Chatbot",
    content: "WanderLust is a feature-rich Airbnb-inspired travel stay discovery platform. Architecture: Node.js Express MVC pattern EJS templating with ejs-mate for layouts MongoDB Atlas as cloud database. app.js mounts 9 routers: listingRouter for property CRUD, reviewsRouter for user reviews and ratings, userRouter for authentication, chatbotRouter for AI property search chat, bookingRouter for stay reservations, paymentRouter for Stripe payment integration, wishlistRouter for saving favorite listings, dashboardRouter for user dashboard, comparisonRouter for comparing properties side by side. Authentication: Passport.js with passport-local strategy express-session with MongoStore for session persistence in MongoDB. Security: Helmet.js with custom CSP directives allowing jsdelivr cloudflare CDNs js.stripe.com express-mongo-sanitize preventing NoSQL injection compression gzip. Models: Listing property details location images, Review user reviews and star ratings, User with authentication, Booking reservation calendar, Wishlist saved properties. Cloudinary for image storage and transformation via cloudConfig.js. MVC Controllers: listingController reviewController userController. Validation: Joi schemas listingSchema reviewSchema in schema.js with custom ExpressError class and wrapAsync async error handler. Features: property listing with MapBox geospatial map integration, user authentication and authorization, review system, booking calendar, Stripe payment processing, wishlist, property comparison side by side, AI chatbot for property search assistance, geospatial search. Docker and docker-compose. MongoDB Atlas cloud database. GitHub: https://github.com/PAMIDIROHIT/Wanderlust",
    keywords: ["wanderlust", "travel", "booking", "stay", "accommodation", "hotel", "airbnb", "mongodb", "nodejs", "express", "passport", "cloudinary", "stripe", "helmet", "ejs", "mvc", "chatbot", "wishlist", "comparison", "mapbox", "geospatial", "reviews", "listings", "fullstack", "atlas", "docker", "joi", "validation"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: AI SEARCH CHAT PDF
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-ai-chat-pdf",
    category: "project",
    title: "AI Search Chat PDF Viewer — Perplexity-Style SSE Streaming Next.js 16",
    content: "An AI-powered search and chat application inspired by Perplexity AI with integrated PDF capabilities. Architecture: Next.js 16 TypeScript frontend plus FastAPI Python backend. Core features: Real-time SSE Server-Sent Events streaming for AI responses token by token, split-view PDF viewer with inline citations and annotations, dark mode with generative UI components, Tailwind CSS styling with full TypeScript. The application allows users to ask questions and receive streaming AI responses with source citations, view cited PDFs side-by-side with the answer, and create a seamless research workflow. Built with Next.js 16 App Router FastAPI streaming endpoints Tailwind CSS and TypeScript. Demonstrates expertise in streaming architectures PDF handling and modern AI chat UX patterns. GitHub: https://github.com/PAMIDIROHIT/AI-search-chat-pdf-viewer",
    keywords: ["ai search", "chat pdf", "perplexity", "sse", "streaming", "server-sent events", "nextjs", "next.js 16", "fastapi", "typescript", "generative ui", "dark mode", "citations", "pdf viewer", "ai chat", "tailwind", "split view", "research"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: KNOWLEDGE GRAPH RAG
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-knowledge-graph",
    category: "project",
    title: "Knowledge Graph Augmented Retrieval System — Advanced RAG Architecture",
    content: "Knowledge Graph Augmented Retrieval System KGARS is a Python research project that enhances RAG Retrieval-Augmented Generation with structured knowledge graph representations. Traditional RAG retrieves flat document chunks. KGARS builds a knowledge graph where entities nodes and relationships edges are extracted from documents enabling relationship-aware context retrieval. When answering a question the system traverses the graph to gather not just the most similar documents but also related entities across multiple hops providing richer more connected context. Demonstrates understanding of advanced RAG architectures graph databases compatible with Neo4j Python NLP pipelines spaCy entity extraction and information retrieval. GitHub: https://github.com/PAMIDIROHIT/-Knowledge-Graph-Augmented-Retrieval-System",
    keywords: ["knowledge graph", "rag", "retrieval augmented generation", "graph", "augmented retrieval", "python", "nlp", "information retrieval", "ai", "neo4j", "entities", "relationships", "graph traversal", "kgars", "advanced rag", "multiagent", "multi-agent", "agent"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: QUICKCART
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-quickcart",
    category: "project",
    title: "QuickCart — Smart Fast E-Commerce Web Application",
    content: "QuickCart is a modern e-commerce web application focused on speed and user experience. Features: product browsing with category filters smart cart management add remove update quantities seamless checkout flow product search user authentication order management. Tech stack: JavaScript Node.js backend clean UI with focus on performance. Built with security-first approach input validation secure sessions XSS prevention. Demonstrates full e-commerce workflow from product discovery to order completion. GitHub: https://github.com/PAMIDIROHIT/QuickCart-main",
    keywords: ["quickcart", "ecommerce", "shopping", "cart", "checkout", "online store", "products", "shop", "buy", "retail", "nodejs", "javascript", "orders"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: BEYONDCHATS
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-beyondchats",
    category: "project",
    title: "BeyondChats — AI-Powered Article Optimization Platform",
    content: "BeyondChats is a full-stack AI article optimization platform. Node.js Express backend with MongoDB Docker containerized. server.js sets up Helmet security headers CORS configuration rate limiting 100 requests per 15 minutes per IP via express-rate-limit body parsers with 10mb limit. Backend modules: config for database connection, controllers for article CRUD and AI optimization, models for article schema, routes REST endpoints for articles, services for AI analysis service that analyzes article content suggests SEO improvements rewrites sections and optimizes for engagement, scripts for database seeding. Frontend full-stack UI for article management AI optimization workflow and real-time preview. Documentation includes API reference DOCUMENTATION.md MongoDB setup MONGODB_SETUP.md GITHUB_SETUP.md TESTING_RESULTS.md. Docker-compose for full stack deployment. GitHub: https://github.com/PAMIDIROHIT/beyondchats-assignment",
    keywords: ["beyondchats", "article", "content optimization", "seo", "ai", "express", "mongodb", "nodejs", "rate limiting", "helmet", "docker", "article analysis", "content", "optimization", "ai article"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: BRINAVV
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-brinavv",
    category: "project",
    title: "Brinavv — MERN Stack Task Management System",
    content: "Brinavv is a full-stack MERN MongoDB Express React Node.js task management system. Features complete task CRUD Create Read Update Delete user authentication with JWT task assignment status tracking To-Do In-Progress Done priority levels deadline management and team collaboration features. Built with JavaScript throughout the stack. Clean separation between React frontend and Express backend. MongoDB for data persistence. JWT-based authentication for secure API endpoints. GitHub: https://github.com/PAMIDIROHIT/Brinavv_Assignment",
    keywords: ["brinavv", "task management", "mern", "mongodb", "express", "react", "nodejs", "jwt", "crud", "todos", "assignment", "collaboration", "team"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: TASKFLOW
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-taskflow",
    category: "project",
    title: "TaskFlow Backend — Node.js REST API Assignment",
    content: "TaskFlow is a Node.js backend REST API built as an assignment project focused on clean architecture and RESTful design. JavaScript-based Express API with task management endpoints user authentication and structured data handling. Demonstrates solid understanding of backend development patterns middleware composition and API design principles including proper HTTP methods status codes and error handling. GitHub: https://github.com/PAMIDIROHIT/taskflow-backend-assignment",
    keywords: ["taskflow", "backend", "rest api", "nodejs", "express", "javascript", "middleware", "task", "assignment"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: MATERNAL HEALTH
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-maternal-health",
    category: "project",
    title: "Maternal Health Risk Prediction — RandomForest Pregnancy Risk Classification",
    content: "Maternal Health Risk Prediction is an AI system to assess pregnancy-related health risks using machine learning. Built with Python Jupyter Notebook and scikit-learn. Uses a pre-trained RandomForest classifier trained on maternal health dataset. Features: age systolic BP diastolic BP blood sugar body temperature heart rate. Outputs risk level: Low Risk Mid Risk or High Risk with confidence scores. The model analyzes user-provided health inputs and generates reliable risk predictions with clear recommendations. Jupyter notebooks demonstrate exploratory data analysis feature correlation heatmaps model selection comparing multiple classifiers and evaluation metrics. Provides clear health insights and smart recommendations for improved maternal care. GitHub: https://github.com/PAMIDIROHIT/Maternal_Health_Risk",
    keywords: ["maternal", "health", "pregnancy", "risk prediction", "random forest", "healthcare", "ai", "medical", "mother", "prenatal", "blood pressure", "blood sugar", "python", "scikit-learn", "jupyter", "classification"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: RL TRAFFIC CONTROL
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-rl-traffic",
    category: "project",
    title: "RL Traffic Control System — Deep Q-Network for Smart Traffic Signals",
    content: "A Reinforcement Learning-based intelligent traffic signal control system. The RL agent uses Deep Q-Network DQN or Q-Learning to learn optimal signal timing policies at intersections. State space: queue lengths at each lane time since last phase change current phase. Action space: next traffic signal phase to activate. Reward function minimizes total vehicle waiting time and maximizes throughput. The RL agent replaces fixed-timing controllers adapting dynamically to real-time traffic patterns. Built with Python using RL libraries. Demonstrates expertise in Markov Decision Processes Q-learning reward shaping and traffic simulation. The system reduces average vehicle waiting time across simulated intersections through learned adaptive policies. GitHub: https://github.com/PAMIDIROHIT/RL and https://github.com/PAMIDIROHIT/RL_TRAFFIC_PER",
    keywords: ["reinforcement learning", "rl", "traffic", "control", "optimization", "ai", "signals", "intersection", "autonomous", "dqn", "q-learning", "agent", "markov", "reward", "python", "deep learning", "rl_traffic_per", "rlunit2"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: TELUGU VAE BTP
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-telugu-vae",
    category: "project",
    title: "Telugu VAE BTP — Variational Autoencoder for Low-Resource NLP Research",
    content: "B.Tech Project BTP research on Variational Autoencoder VAE for Telugu language processing. Telugu is a Dravidian language spoken by 75 plus million people and severely underrepresented in NLP research. The project trains a VAE on Telugu text data to learn latent representations of Telugu sentences enabling tasks like text generation in Telugu semantic similarity in Telugu and data augmentation for low-resource NLP downstream tasks. Variational Autoencoders learn a continuous latent space where similar texts cluster together enabling interpolation and novel sample generation. Built with Python Jupyter Notebook PyTorch or TensorFlow. Demonstrates deep research capability in low-resource NLP generative models variational inference and regional language AI. GitHub: https://github.com/PAMIDIROHIT/telugu-vae-btp",
    keywords: ["btp", "b.tech project", "telugu", "vae", "variational autoencoder", "nlp", "research", "regional language", "language model", "dravidian", "text generation", "jupyter", "pytorch", "tensorflow", "generative", "latent space"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: HUMAN ALIGNED AI
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-human-aligned-ai",
    category: "project",
    title: "Human-Aligned AI — AI Safety and Alignment Research System",
    content: "Human-Aligned AI is a Python research project focused on aligning AI systems with human values and intentions. The field of AI alignment addresses making AI systems safer more interpretable explainable and compliant with human preferences. Techniques explored include RLHF Reinforcement Learning from Human Feedback Constitutional AI principles preference learning from human demonstrations and interpretability methods for understanding model decisions. The project demonstrates Rohit's awareness of responsible AI development beyond just building capable systems understanding the critical importance of safe and value-aligned AI. GitHub: https://github.com/PAMIDIROHIT/Human-Aligned-AI-",
    keywords: ["human aligned", "ai alignment", "safe ai", "python", "alignment", "rlhf", "ethics", "safety", "interpretability", "explainability", "preference learning", "responsible ai", "constitutional ai"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: TAILORTALK
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-tailortalk",
    category: "project",
    title: "TailorTalk — Personalized AI Conversation and Chat System",
    content: "TailorTalk is a Python-based AI conversation system focused on delivering personalized context-aware chat experiences. The system tailors responses based on user context conversation history and inferred user preferences going well beyond generic chatbot responses. Demonstrates knowledge of dialogue systems context window management NLP pipelines and personalization strategies for better user experience. Built with Python leveraging NLP libraries and conversational AI patterns. GitHub: https://github.com/PAMIDIROHIT/Tailortalk",
    keywords: ["tailortalk", "chatbot", "conversation", "ai chat", "python", "personalized", "nlp", "context-aware", "dialogue", "preferences", "chat system"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: TEXT SUMMARIZATION
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-text-summarization",
    category: "project",
    title: "Text Summarization — Extractive and Abstractive NLP Summarizer",
    content: "Text Summarization project in Jupyter Notebook implementing both extractive summarization selecting key sentences using TF-IDF importance scoring and TextRank graph algorithm and abstractive summarization using transformer models to generate novel summaries. Explores NLP techniques including tokenization sentence scoring cosine similarity for sentence selection and transformer fine-tuning. Built with Python NLTK Hugging Face Transformers and scikit-learn. Demonstrates NLP research skills and understanding of modern summarization approaches for both extractive graph-based and neural generative methods. GitHub: https://github.com/PAMIDIROHIT/Text_Summarization",
    keywords: ["text summarization", "nlp", "extractive", "abstractive", "tfidf", "textrank", "transformer", "hugging face", "python", "nltk", "jupyter", "summary", "summarization"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: HEALTHCARE DISEASE PREDICTION
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-healthcare-prediction",
    category: "project",
    title: "Healthcare Disease Prediction Risk Assessment — Clinical ML System",
    content: "A comprehensive ML system for predicting disease risk from clinical parameters. Built with Python and Jupyter Notebook uses scikit-learn for model development. Focuses on disease prediction likelihood using patient clinical data. ML pipeline: data loading and EDA exploratory data analysis feature engineering model training with classification algorithms cross-validation performance evaluation and risk assessment output. The system enables early intervention and preventive healthcare decisions by clinicians. Demonstrates end-to-end ML workflow for medical applications with attention to data quality model reliability and clinical interpretability. GitHub: https://github.com/PAMIDIROHIT/Healthcare-Disease-Prediction-Risk-Assessment-System",
    keywords: ["healthcare", "disease prediction", "risk assessment", "clinical", "jupyter", "scikit-learn", "python", "heart disease", "ml pipeline", "early detection", "preventive healthcare"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: CODSOFT
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-codsoft",
    category: "project",
    title: "CODSOFT — Web Development Projects and Internship Tasks",
    content: "CODSOFT is a collection of HTML CSS JavaScript web development projects built as part of the CODSOFT internship task program. Features clean frontend implementations demonstrating proficiency in HTML5 semantic markup CSS3 animations and layouts vanilla JavaScript DOM manipulation and responsive web design with media queries. The projects showcase fundamental web development skills and attention to UI/UX detail including landing pages and interactive web components. GitHub: https://github.com/PAMIDIROHIT/CODSOFT",
    keywords: ["codsoft", "html", "css", "javascript", "web development", "frontend", "internship", "web projects", "responsive design"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: BAH25 LUNAR PIONEERS (Hackathon)
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-bah25-lunar",
    category: "project",
    title: "Lunar Hazard Detection System — BAH25 Chandrayaan AI for Moon Surface Analysis",
    content: "AI-powered Lunar Hazard Detection System built for Bharatiya Antariksh Hackathon 2025 BAH25 Problem Statement PS-11: Novel method to detect landslides and boulders on the Moon using Chandrayaan images. Team: Lunar Pioneers. Rohit was Team Member-3 from IIIT Sri City. The system combines multi-modal sensor fusion from three Chandrayaan instruments: TMC Terrain Mapping Camera for 3D surface mapping, DTM Digital Terrain Model for elevation profiles, and OHRC Orbiter High Resolution Camera for sub-meter resolution imaging. Core AI Architecture: Physics-Informed Neural Network PINN that encodes lunar geological constraints including lunar gravity 1.62 m/s2 regolith cohesion properties and slope stability thresholds into the loss function. Attention-based sensor fusion module uses pixel-wise cross-attention between TMC DTM and OHRC feature maps to learn complementary information. Landslide Detection: Identifies surface deformation patterns slope instabilities and mass wasting signatures. Boulder Detection: Uses shadow geometry analysis computing shadow length and sun angle to estimate boulder height and volume. Risk Classification: Multi-class output categorizing regions as High Medium or Low risk with calibrated uncertainty scores. Temporal Change Detection: Compares data across Chandrayaan-1 and Chandrayaan-2 missions to detect new hazards. Web Interface: React.js TypeScript frontend with interactive lunar surface maps heat-map risk overlays and exportable reports. Backend: MongoDB for storing analysis results and session data. Training: Google Colab and Kaggle GPU instances. Tech Stack: PyTorch TorchVision scikit-learn OpenCV Rasterio GDAL NumPy SciPy React.js TypeScript MongoDB. 100 percent open-source with zero financial cost. USP: First system combining TMC plus DTM plus OHRC with pixel-wise attention and physics-informed AI using lunar gravity and regolith dynamics.",
    keywords: ["bah25", "lunar pioneers", "moon", "chandrayaan", "isro", "hackathon", "landslide", "boulder", "physics-informed neural network", "sensor fusion", "tmc", "dtm", "ohrc", "pytorch", "opencv", "rasterio", "gdal", "react", "typescript", "mongodb", "lunar hazard", "space", "ai", "surface analysis", "risk classification", "attention mechanism", "ps-11"]
  },

  /* ═══════════════════════════════════════════════════════
     PROJECT: NEUROCORE AGENTIC AI (KernelCrew)
  ═══════════════════════════════════════════════════════ */
  {
    id: "project-neurocore-agentic",
    category: "project",
    title: "NeuroCore — Multi-Agent AI Smart City Platform with 6 Specialized Agents",
    content: "NeuroCore is an AI-powered smart-city management platform built by Team KernelCrew for the Agentic AI competition. Problem Statement: Managing City Data Overload. NeuroCore deploys 6 specialized AI agents that collaborate through a Consensus Engine to make unified city-wide decisions. The 6 Agents: 1. Traffic Agent monitors real-time traffic flow predicts congestion and optimizes signal timing using Google Maps API and Vertex AI Forecasting. 2. Safety Agent processes surveillance feeds with Vertex AI Vision for anomaly detection and automated threat response. 3. Health Agent tracks disease patterns manages hospital capacity and triggers pandemic early warnings. 4. Environment Agent monitors air quality water levels noise pollution and provides sustainability metrics. 5. Emergency Agent coordinates disaster response allocates resources and runs evacuation simulations. 6. Social Agent analyzes citizen sentiment from social media performs emotion geography mapping and gauges public service satisfaction. Key Innovations: Consensus Engine enables cross-domain multi-agent collaboration where agents vote and negotiate on city-wide decisions. Neural Event Mesh provides real-time event propagation across all agents. Quantum Route Intelligence optimizes city-wide routing beyond simple shortest-path. Emotion Geography maps citizen sentiment across city zones. City Oracle allows natural language what-if simulations. Smart Crisis Simulation enables preemptive disaster planning. Privacy-by-Design architecture with Edge AI processing sensitive data locally federated learning and differential privacy for anonymization. Tech Stack: Gemini 1.5 Pro for core reasoning, Vertex AI Vision Agent Builder and Forecasting, Google Maps API, React plus TypeScript frontend, Node.js plus Express plus MongoDB backend, Firebase Firestore Studio and Hosting for real-time data, Edge AI for privacy-preserving local processing. Two deployment modes: Full deployment for metropolitan cities and Lite deployment for smaller municipalities. USP: First agentic architecture for smart cities with cross-domain multi-agent collaboration and consensus-based decision making.",
    keywords: ["neurocore", "agentic ai", "smart city", "multi-agent", "kernelcrew", "kernel crew", "consensus engine", "traffic agent", "safety agent", "health agent", "environment agent", "emergency agent", "social agent", "gemini", "vertex ai", "google maps", "firebase", "edge ai", "city oracle", "emotion geography", "neural event mesh", "react", "typescript", "mongodb", "nodejs", "express", "privacy", "federated learning", "hackathon", "competition"]
  },

  /* ═══════════════════════════════════════════════════════
     SKILLS — DETAILED BREAKDOWNS
  ═══════════════════════════════════════════════════════ */
  {
    id: "skills-languages",
    category: "skills",
    title: "Programming Languages — C++ Python TypeScript JavaScript Java C",
    content: "Rohit is proficient in: C++ for competitive programming DSA algorithms and LeetCode 600+ problems, C for systems programming fundamentals, Java for OOP, Python as primary ML/AI language used in CineMatch HeartDisease Maternal Health RL Traffic Telugu VAE Text Summarization KGARS Human-Aligned AI TailorTalk, JavaScript for Node.js backends used in Tomato Wanderlust QuickCart TaskFlow BeyondChats Brinavv, TypeScript as preferred production language used in NestJS Next.js CineMatch AI Chat PDF. TypeScript is his language of choice for large-scale enterprise applications.",
    keywords: ["programming languages", "c++", "c", "java", "python", "javascript", "typescript", "coding", "languages", "competitive programming"]
  },
  {
    id: "skills-frontend",
    category: "skills",
    title: "Frontend — React Next.js 14/16 Redux TypeScript Tailwind CSS Chart.js",
    content: "Rohit frontend tech stack: React.js used in Tomato 3-panel food delivery with Vite HeartDisease prediction UI with Chart.js visualizations, Next.js 14 in CineMatch AI with App Router SSR SSG, Next.js 16 in AI Search Chat PDF with SSE streaming, Redux for state management, Tailwind CSS in CineMatch and AI Chat PDF, TypeScript in CineMatch AI Chat PDF and NestJS frontend, HTML5 CSS3 in CODSOFT projects, Chart.js for data visualization in medical apps. Experienced in all Next.js rendering patterns CSR SSR SSG ISR. Builds responsive accessible and performant user interfaces.",
    keywords: ["frontend", "react", "reactjs", "next.js", "nextjs", "redux", "html", "css", "tailwind", "ui", "interface", "web", "design", "responsive", "chart.js", "vite", "ssr", "ssg", "app router"]
  },
  {
    id: "skills-backend",
    category: "skills",
    title: "Backend — Node.js NestJS Express.js Flask FastAPI GraphQL REST APIs",
    content: "Rohit backend expertise: Node.js Express.js used in Tomato most complex app with Socket.io Passport OAuth multer Redis real-time WebSockets and 9 API route groups, Wanderlust with 9 routers including Stripe Cloudinary Helmet, BeyondChats QuickCart TaskFlow Brinavv and NeuroCore smart city backend. NestJS for production REST API with 51 unit tests TypeORM JWT guards validation pipes global exception filters Helmet module-based architecture. Flask for HeartDisease ML API and CineMatch AI Python backend with Flask-Caching and Flask-Limiter. FastAPI for AI Search Chat PDF with streaming SSE endpoints. GraphQL experience for flexible API data fetching alongside REST. Advanced backend skills: middleware composition authentication flows WebSocket real-time communication rate limiting file upload handling and API security hardening.",
    keywords: ["backend", "nodejs", "node", "express", "expressjs", "nestjs", "flask", "fastapi", "graphql", "restful", "api", "server", "microservices", "middleware", "socket.io", "websocket", "passport", "multer", "rate limiting"]
  },
  {
    id: "skills-database",
    category: "skills",
    title: "Databases — MongoDB PostgreSQL Redis MySQL Supabase Atlas",
    content: "Database expertise: MongoDB with Mongoose ODM used in Tomato with 6 models conversationModel deliveryPartnerModel orderModel foodModel userModel notificationModel, Wanderlust with MongoDB Atlas cloud and MongoStore sessions, CineMatch for watchlists, BeyondChats beyondchats. PostgreSQL via Supabase used in NestJS API with TypeORM entities and migrations schema management. Redis used in Tomato caching achieving 81 percent API response time reduction from 800ms to 150ms, CineMatch Flask-Caching with Redis URL backend for 300-600s movie caching. MySQL for classical relational database projects. Supabase cloud PostgreSQL fully managed. MongoDB Atlas cloud hosted. Understands: schema design indexing query optimization NoSQL document modeling relational normalization session persistence different storage backends.",
    keywords: ["database", "mysql", "mongodb", "redis", "sql", "nosql", "storage", "cache", "postgres", "postgresql", "supabase", "db", "mongoose", "typeorm", "atlas", "cloud database", "indexing", "mongostore"]
  },
  {
    id: "skills-ml-ai",
    category: "skills",
    title: "ML/AI — scikit-learn XGBoost NLP RAG LangChain LangGraph Gemini RL TF-IDF PyTorch",
    content: "Rohit ML and AI toolkit: scikit-learn for RandomForest LogisticRegression DecisionTree SVM MLP all in HeartDisease GridSearchCV cross_val_score evaluation metrics RandomForest in Maternal Health, XGBoost for HeartDisease model comparison achieving superior accuracy, pandas and numpy for CineMatch data processing HeartDisease feature engineering, TF-IDF plus Cosine Similarity for CineMatch content-based filtering with pickle-saved precomputed similarity matrix on 10000 plus movies, RAG built from scratch for portfolio chatbot TF-IDF indexing cosine similarity retrieval intent detection template-based generation, Knowledge Graph RAG in KGARS project, LangChain for recommendation pipelines, LangGraph for building stateful multi-agent agentic AI workflows with graph-based orchestration, Reinforcement Learning Q-Learning DQN for RL Traffic Control with custom reward functions, VAE Variational Autoencoders for Telugu NLP in BTP research, RLHF concepts in Human-Aligned AI, Gemini AI API google/generative-ai SDK gemini-1.5-flash model for Tomato food delivery chatbot with live menu context caching and Gemini 1.5 Pro in NeuroCore smart city agents, PyTorch and TorchVision for BAH25 Lunar Hazard Detection physics-informed neural network with attention-based sensor fusion, Text Summarization extractive TextRank and abstractive Transformer-based, NLP tokenization embeddings stopwords stemming TF-IDF, OpenCV Rasterio GDAL for geospatial image processing in lunar surface analysis. Evaluation: accuracy precision recall F1 roc_auc confusion matrix cross-validation.",
    keywords: ["machine learning", "ml", "ai", "artificial intelligence", "deep learning", "nlp", "natural language processing", "langchain", "rag", "scikit", "neural network", "embedding", "xgboost", "random forest", "tfidf", "cosine similarity", "reinforcement learning", "vae", "variational autoencoder", "gemini", "hugging face", "transformers", "gridsearchcv"]
  },
  {
    id: "skills-devops",
    category: "skills",
    title: "DevOps — Docker Docker-Compose AWS EC2 CI/CD Git GitHub Databricks",
    content: "DevOps and cloud skills: Docker used in all production projects including Tomato multi-container with frontend backend admin, Wanderlust, NestJS API, HeartDisease on AWS EC2, CineMatch, BeyondChats. Docker Compose for multi-service orchestration in Wanderlust NestJS BeyondChats defining service networks volumes and dependencies. AWS EC2 deployment for HeartDisease ML with 40 percent latency improvement through code optimization and instance tuning. CI/CD pipelines in Tomato food delivery for automated testing and deployment. Git and GitHub for 22 plus repositories with structured commits feature branches and documentation. Databricks for big data processing ML workflows. Makefile for build automation in HeartDisease. Containerizes all production projects as standard practice.",
    keywords: ["docker", "aws", "amazon web services", "ec2", "git", "github", "ci/cd", "cloud", "devops", "databricks", "tools", "deployment", "containerization", "cicd", "docker-compose", "multi-container"]
  },

  /* ═══════════════════════════════════════════════════════
     SKILLS: MLOPS & MULTI-AGENT SYSTEMS
  ═══════════════════════════════════════════════════════ */
  {
    id: "skills-mlops",
    category: "skills",
    title: "MLOps & Multi-Agent Systems — Model Deployment Pipelines and Agent Architectures",
    content: "Rohit has practical MLOps experience gained through deploying Generative AI and ML projects end-to-end. MLOps skills include: Docker containerization of ML models for HeartDisease Flask API CineMatch backend and Maternal Health services, CI/CD pipelines for automated model training and deployment in Tomato food delivery, model versioning with pickle serialized similarity matrices in CineMatch, API serving of ML models via Flask endpoints and FastAPI streaming endpoints, experiment tracking across 6 competing ML models with GridSearchCV cross-validation in HeartDisease, feature engineering pipelines with pandas and scikit-learn preprocessing StandardScaler, cloud deployment on AWS EC2 with performance monitoring and 40 percent latency optimization, Makefile-based build automation for reproducible ML workflows. Multi-Agent Systems: Rohit demonstrates multi-agent architecture understanding through multiple projects. NeuroCore smart city platform deploys 6 specialized AI agents Traffic Safety Health Environment Emergency Social with a Consensus Engine for cross-domain collaboration using Gemini 1.5 Pro and Vertex AI. Knowledge Graph Augmented Retrieval System KGARS uses specialized agents for graph traversal entity extraction and context assembly working together as a multi-agent pipeline. Tomato Food Delivery has independent microservices admin panel customer frontend backend server chatbot service recommendation engine acting as autonomous agents communicating via Socket.io WebSockets. Human-Aligned AI explores multi-agent alignment RLHF and Constitutional AI for coordinating multiple AI agents safely. LangGraph enables building stateful agentic workflows with graph-based orchestration. The portfolio RAG chatbot itself is an agentic system with intent detection routing to specialized response generator agents.",
    keywords: ["mlops", "ml ops", "model deployment", "model serving", "multi-agent", "multiagent", "agent", "agents", "pipeline", "experiment tracking", "model versioning", "ci/cd", "docker ml", "agentic", "autonomous agents", "model training", "feature engineering", "ml pipeline", "deployment pipeline"]
  },

  /* ═══════════════════════════════════════════════════════
     ACHIEVEMENTS
  ═══════════════════════════════════════════════════════ */
  {
    id: "achievement-leetcode",
    category: "achievement",
    title: "LeetCode 600+ Problems Solved — Algorithms and DSA",
    content: "Pamidi Rohit has solved 600 plus problems on LeetCode demonstrating strong algorithmic problem-solving skills. Topics mastered: Arrays and Strings using sliding window and two pointers, Linked Lists, Trees with DFS BFS and balanced BST invariants, Graphs using Dijkstra BFS shortest path and union-find, Dynamic Programming 1D and 2D DP knapsack LCS edit distance, Backtracking N-Queens subsets permutations, Binary Search lower upper bound search in rotated array, Heaps and Priority Queues, Greedy algorithms, Stack and Queue problems, Bit manipulation. These skills directly support his work in writing efficient backend algorithms. LeetCode: https://leetcode.com/u/rohithtnsp/",
    keywords: ["leetcode", "competitive programming", "algorithms", "data structures", "dsa", "problem solving", "600 problems", "coding", "interview prep", "competitive", "dynamic programming", "graphs", "trees", "binary search"]
  },
  {
    id: "achievement-hackathon",
    category: "achievement",
    title: "Bharatiya Antariksh Hackathon 2025 — Lunar Pioneers Team, ISRO PS-11",
    content: "Rohit participated in the Bharatiya Antariksh Hackathon 2025 BAH25 a prestigious national-level hackathon by ISRO under the Indian Space Research Organisation. Team Name: Lunar Pioneers. Problem Statement: PS-11 Novel method to detect landslides and boulders on the Moon using Chandrayaan images. Rohit was a core team member from IIIT Sri City. The team built an AI-powered Lunar Hazard Detection System combining multi-modal sensor data from Chandrayaan missions including TMC Terrain Mapping Camera, DTM Digital Terrain Model, and OHRC Orbiter High Resolution Camera. The solution features a Physics-Informed Neural Network trained on lunar geological principles, attention-based sensor fusion for pixel-wise hazard classification, landslide detection via surface deformation analysis, boulder detection through shadow geometry analysis, risk classification into High Medium Low zones with uncertainty scoring, and temporal change detection across Chandrayaan-1 and Chandrayaan-2 missions. Tech stack: PyTorch, TorchVision, scikit-learn, OpenCV, Rasterio, GDAL, NumPy, SciPy, React.js, TypeScript, MongoDB, Google Colab, Kaggle. The entire solution is 100 percent open-source with no financial cost. USP: First system combining TMC plus DTM plus OHRC with pixel-wise attention and physics-informed AI leveraging lunar gravity and regolith dynamics.",
    keywords: ["hackathon", "bharatiya antariksh", "bah25", "bah", "space", "isro", "lunar pioneers", "moon", "chandrayaan", "landslide", "boulder", "physics-informed", "neural network", "sensor fusion", "tmc", "dtm", "ohrc", "pytorch", "opencv", "national hackathon", "2025", "ps-11", "lunar", "hazard detection", "competition"]
  },

  /* ═══════════════════════════════════════════════════════
     CONVERSATIONAL & CONTEXTUAL RESPONSES
  ═══════════════════════════════════════════════════════ */
  {
    id: "conv-rohit-strengths",
    category: "conversational",
    title: "Why Hire Rohit — Strengths and Value Proposition",
    content: "Rohit is an exceptional hire because: 1. Full-stack versatility — he can own frontend (React, Next.js), backend (Node.js, NestJS, Flask, FastAPI), databases (MongoDB, PostgreSQL, Redis), and ML pipelines end-to-end. 2. Proven builder — 22+ real-world projects, not just tutorials. He has built production food delivery platforms, movie recommendation engines, enterprise APIs with 51 tests, and medical ML systems deployed on AWS. 3. Strong foundations — 600+ LeetCode problems prove deep algorithmic thinking. 4. AI/ML depth — from classical ML (scikit-learn, XGBoost) to deep learning (PyTorch, physics-informed neural networks), NLP (TF-IDF, VAE for Telugu), and cutting-edge agentic AI (LangGraph, multi-agent systems). 5. Research capability — B.Tech project on low-resource NLP, ISRO hackathon finalist, AI alignment research. 6. DevOps-ready — Docker, CI/CD, AWS deployment experience. He is not a student who just follows tutorials — he architects, builds, and deploys real systems.",
    keywords: ["hire", "recruit", "why hire", "strengths", "value", "good", "best", "impressive", "talented", "capable", "worth", "should i hire", "strong candidate", "team", "fit"]
  },
  {
    id: "conv-rohit-availability",
    category: "conversational",
    title: "Rohit Availability and Job Interests",
    content: "Rohit is actively seeking internships and full-time positions. He is available for: Software Engineering roles, ML/AI Engineering roles, Full-Stack Development roles, Backend Development roles, Data Science roles, and Research positions. He is also open to freelance projects and open-source collaboration. Expected graduation: May 2027 from IIIT Sri City. Preferred locations: Open to remote, hybrid, or on-site opportunities across India and internationally. Contact him at rohithtnsp@gmail.com or +91 9398026237.",
    keywords: ["available", "availability", "looking for", "job", "internship", "opportunity", "open to", "position", "role", "work with", "freelance", "remote", "hire"]
  },
  {
    id: "conv-rohit-experience-level",
    category: "conversational",
    title: "Rohit Experience Level and Work Readiness",
    content: "Rohit is a B.Tech student (graduating May 2027) with extensive practical experience. While still in college, his project portfolio demonstrates senior-level engineering skills: production-grade microservices with WebSocket real-time communication, enterprise NestJS APIs with 51 unit tests, ML model deployment on AWS EC2 with 40% latency optimization, Docker containerization as standard practice, real-time payment integration with Razorpay and Stripe, OAuth authentication with Passport.js, and physics-informed neural networks for ISRO. He has the depth of a working professional combined with the fresh perspective and energy of a motivated student.",
    keywords: ["experience", "level", "junior", "senior", "intern", "fresher", "years of experience", "work experience", "professional", "ready", "qualified"]
  },
  {
    id: "conv-rohit-personality",
    category: "conversational",
    title: "Rohit Personality and Working Style",
    content: "Rohit is a passionate builder who loves solving complex problems. He is self-driven — his 22+ projects were built out of genuine curiosity, not classroom assignments. He is a fast learner who picks up new technologies quickly (evidenced by his breadth from React to PyTorch to LangGraph). He is detail-oriented — his NestJS API has 51 unit tests, his HeartDisease project compares 6 ML models with cross-validation, and his hackathon submission uses physics-informed neural networks. He collaborates well in teams — he was part of Team Lunar Pioneers (BAH25) and Team KernelCrew (NeuroCore). He is also an effective communicator — his project documentation and portfolio demonstrate clear technical writing.",
    keywords: ["personality", "working style", "team player", "work ethic", "motivated", "passionate", "driven", "character", "person", "like", "type"]
  },
  {
    id: "conv-fun-facts",
    category: "conversational",
    title: "Fun Facts and Interesting Things About Rohit",
    content: "Fun facts about Pamidi Rohit: 1. He has solved 600+ LeetCode problems — that is more than most professional developers. 2. He built a physics-informed neural network that understands lunar gravity (1.62 m/s²) for ISRO. 3. His food delivery app Tomato has a Gemini AI chatbot that knows the entire restaurant menu in real-time. 4. He reduced API response time by 81% using Redis caching (800ms to 150ms). 5. He is researching Variational Autoencoders for Telugu — a language spoken by 75+ million people but underrepresented in NLP. 6. His NeuroCore platform has 6 AI agents that can vote and negotiate to make city-wide decisions. 7. He containerizes ALL his production projects with Docker as standard practice. 8. He studies at IIIT Sri City — an institute of national importance under the Ministry of Education. 9. He scored 93.7% in his intermediate exams. 10. This very chatbot you are using is a RAG system he built from scratch using TF-IDF and cosine similarity.",
    keywords: ["fun fact", "interesting", "cool", "amazing", "wow", "surprising", "did you know", "random", "trivia", "unique"]
  },

  /* ═══════════════════════════════════════════════════════
     SUMMARIZED OVERVIEWS
  ═══════════════════════════════════════════════════════ */
  {
    id: "projects-overview",
    category: "project",
    title: "Complete Portfolio — All 22 Projects by Pamidi Rohit",
    content: "Pamidi Rohit complete GitHub portfolio with 22 plus repositories: 1. Tomato Food Delivery Node React MongoDB Redis Gemini AI Socket.io production grade 3-panel platform. 2. CineMatch AI Flask Next.js TF-IDF cosine similarity 10K movies 95% accuracy. 3. NestJS API TypeScript PostgreSQL Supabase 51 unit tests enterprise architecture. 4. HeartDisease ML FullStack Python Flask React 6 ML models 90%+ accuracy AWS deployed. 5. WanderLust Node Express MongoDB Cloudinary Stripe Airbnb-like 9 feature routers. 6. AI Search Chat PDF Next.js 16 FastAPI SSE streaming Perplexity-style. 7. Knowledge Graph RAG Python relationship-aware retrieval augmented generation. 8. QuickCart JavaScript Node.js e-commerce platform. 9. Maternal Health Risk Python scikit-learn pregnancy risk classification. 10. RL Traffic Control Python DQN-based intelligent signal optimization. 11. BeyondChats Node Express MongoDB AI article optimization platform. 12. Brinavv MERN task management system. 13. TaskFlow Node Express REST API. 14. Telugu VAE BTP Python Jupyter B.Tech research Dravidian NLP VAE. 15. Human-Aligned AI Python AI safety alignment RLHF. 16. TailorTalk Python personalized conversation AI. 17. Text Summarization Python Jupyter extractive abstractive NLP. 18. Healthcare Disease Prediction Python sklearn clinical risk assessment. 19. CODSOFT HTML CSS JS web development collection. 20. Lunar Hazard Detection System BAH25 PyTorch physics-informed neural network Chandrayaan sensor fusion for ISRO hackathon. 21. NeuroCore multi-agent AI smart city platform with 6 specialized agents Gemini Vertex AI Firebase for Agentic AI competition. 22-24. RL variants additional experiments. GitHub: https://github.com/PAMIDIROHIT",
    keywords: ["all projects", "portfolio", "projects list", "work", "builds", "applications", "repos", "github", "overview", "what has rohit built", "what projects", "22 repos", "22 repositories"]
  },
  {
    id: "skills-overview",
    category: "skills",
    title: "Complete Technical Skills Summary",
    content: "Pamidi Rohit complete tech stack summary. Languages: C++ primary for competitive programming, C, Java, Python primary for ML and AI projects, JavaScript for Node.js backends, TypeScript preferred for production enterprise apps. Frontend: React.js with Vite, Next.js 14 and 16 with App Router SSR SSG ISR, Redux, Tailwind CSS, HTML5, CSS3, Chart.js visualizations. Backend: Node.js plus Express.js primary backend, NestJS for enterprise TypeScript, Flask for ML APIs, FastAPI for streaming SSE, GraphQL for flexible API queries. Databases: MongoDB with Mongoose Atlas cloud, PostgreSQL with TypeORM Supabase cloud, Redis for caching achieving 81% improvement, MySQL. ML and AI: scikit-learn XGBoost PyTorch TF-IDF cosine similarity RAG LangChain LangGraph for agentic workflows Gemini AI API RL Q-Learning DQN VAE NLP OpenCV Rasterio GDAL pandas numpy Vertex AI. DevOps: Docker Docker-Compose AWS EC2 CI/CD Git Databricks Firebase. Architecture: MVC REST Microservices Multi-Agent Systems JWT Auth OAuth WebSockets SSE Streaming. Security patterns: Helmet CSP CORS sanitization rate limiting validation pipes.",
    keywords: ["skills", "technologies", "tech stack", "expertise", "full stack", "what does rohit know", "technologies used", "capabilities", "complete skills", "summary"]
  }
];

/* ═══════════════════════ LOOKUP MAP ═══════════════════════ */
const KB_BY_CATEGORY = {};
KNOWLEDGE_BASE.forEach(doc => {
  if (!KB_BY_CATEGORY[doc.category]) KB_BY_CATEGORY[doc.category] = [];
  KB_BY_CATEGORY[doc.category].push(doc);
});

/* ═══════════════════════ PROFILE METADATA ═══════════════════════ */
const PROFILE = {
  name: "Pamidi Rohit",
  title: "B.Tech AI & Data Science | Full-Stack Developer | ML Engineer",
  college: "IIIT Sri City",
  degree: "B.Tech in Artificial Intelligence and Data Science",
  cgpa: "8.23/10.0",
  graduation: "May 2027",
  email: "rohithtnsp@gmail.com",
  phone: "+91 9398026237",
  whatsapp: "9398026237",
  linkedin: "https://www.linkedin.com/in/rohit-pamidi-4147771ba/",
  github: "https://github.com/PAMIDIROHIT",
  leetcode: "https://leetcode.com/u/rohithtnsp/",
  avatar: "profile.jpg",
  leetcodeProblems: "600+",
  totalRepos: 22,
  topProjects: ["Tomato Food Delivery", "CineMatch AI", "NestJS Production API", "HeartDisease ML FullStack", "WanderLust", "AI Search Chat PDF"],
  skills: {
    languages: ["C++", "C", "Java", "Python", "JavaScript", "TypeScript"],
    frontend: ["React.js", "Next.js 14/16", "Redux", "Tailwind CSS", "HTML5", "CSS3", "Chart.js"],
    backend: ["Node.js", "Express.js", "NestJS", "Flask", "FastAPI", "GraphQL"],
    databases: ["MongoDB", "PostgreSQL", "Redis", "MySQL", "Supabase"],
    mlai: ["scikit-learn", "XGBoost", "PyTorch", "TF-IDF", "RAG", "LangChain", "LangGraph", "Gemini AI", "RL", "VAE", "NLP", "Multi-Agent Systems"],
    tools: ["Docker", "Docker Compose", "AWS EC2", "CI/CD", "Git", "Databricks", "MLOps"]
  }
};

/* ═══════════════════════ PROJECT GITHUB URLS ═══════════════════════ */
const PROJECT_URLS = {
  "project-tomato-food":           "https://github.com/PAMIDIROHIT/Food-Delivery-MAIN1",
  "project-cinematch":             "https://github.com/PAMIDIROHIT/cinematch-ai",
  "project-nestjs-api":            "https://github.com/PAMIDIROHIT/nestjs-api-project",
  "project-heart-disease":         "https://github.com/PAMIDIROHIT/HeartDisease-ML-FullStack",
  "project-wanderlust":            "https://github.com/PAMIDIROHIT/Wanderlust",
  "project-ai-chat-pdf":           "https://github.com/PAMIDIROHIT/AI-search-chat-pdf-viewer",
  "project-knowledge-graph":       "https://github.com/PAMIDIROHIT/-Knowledge-Graph-Augmented-Retrieval-System",
  "project-quickcart":             "https://github.com/PAMIDIROHIT/QuickCart-main",
  "project-maternal-health":       "https://github.com/PAMIDIROHIT/Maternal_Health_Risk",
  "project-rl-traffic":            "https://github.com/PAMIDIROHIT/RL",
  "project-beyondchats":           "https://github.com/PAMIDIROHIT/beyondchats-assignment",
  "project-brinavv":               "https://github.com/PAMIDIROHIT/Brinavv_Assignment",
  "project-taskflow":              "https://github.com/PAMIDIROHIT/taskflow-backend-assignment",
  "project-telugu-vae":            "https://github.com/PAMIDIROHIT/telugu-vae-btp",
  "project-human-aligned-ai":      "https://github.com/PAMIDIROHIT/Human-Aligned-AI-",
  "project-tailortalk":            "https://github.com/PAMIDIROHIT/Tailortalk",
  "project-text-summarization":    "https://github.com/PAMIDIROHIT/Text_Summarization",
  "project-healthcare-prediction": "https://github.com/PAMIDIROHIT/Healthcare-Disease-Prediction-Risk-Assessment-System",
  "project-codsoft":               "https://github.com/PAMIDIROHIT/CODSOFT",
  "project-bah25-lunar":            "https://github.com/PAMIDIROHIT",
  "project-neurocore-agentic":      "https://github.com/PAMIDIROHIT"
};
