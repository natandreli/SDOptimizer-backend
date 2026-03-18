# SDOptimizer Backend

Backend for **SDOptimizer**, a web platform that integrates **System Dynamics** simulation models (Powersim Studio) with **Reinforcement Learning** agents to automate the search for optimal variable configurations — eliminating the need for manual what-if analysis.

## 📋 About the Project

System Dynamics is a key methodology for understanding the behavior of complex systems over time. Tools like **Powersim Studio** are the industry standard for building these models. However, the decision-making process based on them is limited by a critical bottleneck: **manual scenario analysis**.

Currently, optimizing a system (e.g., a supply chain or epidemiological model) requires an analyst to manually adjust input variables, run the simulation, observe results, and repeat — an iterative cycle that is slow, prone to human bias, and unable to explore counterintuitive areas of the solution space.

SDOptimizer addresses this by acting as a bridge between Powersim models (`.sip` files) and an AI optimization engine, automating the full workflow.

## 🧠 How It Works

1. The user uploads a `.mdl` file through the web interface
2. The file passes through a rigorous validation pipeline (size, format, injection patterns)
3. The backend reads and parses the model extracting variables and logic
4. A **Gymnasium-compatible environment** is constructed from the model
5. A **Reinforcement Learning agent** (PPO or SAC) iterates over the simulation, adjusting input parameters to maximize or minimize a user-defined objective function
6. Results are returned comparing the original simulation vs. the optimized configuration

## 🏗️ Project Structure

```
app/
├── api/
│   ├── dependencies/     # Dependency injection
│   └── routers/          # REST API endpoints
├── config/               # App settings (pydantic-settings)
├── core/
│   ├── constants/        # Shared constants
│   ├── operations/       # Business logic
│   ├── validation/       # Validation logic
│   └── utils/            # Helper functions
├── exceptions/           # Custom exception classes
├── lifespan/             # Startup / shutdown events
├── middleware/           # Session management middleware
├── schemas/              # Pydantic request/response models
└── scripts/              # Utility scripts
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Install uv

**On Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/natandreli/SDOptimizer-backend
   cd SDOptimizer-backend
   ```

2. **Create and activate the virtual environment**
   ```bash
   uv venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

### Running the Server

**Development:**
```bash
uv run uvicorn app.main:app --reload
```

**Production:**
```bash
uv run uvicorn app.main:app
```

The API will be available at `http://localhost:8000`.

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### Healthcheck
- `GET /api/healthcheck` — Basic liveness check
- `GET /api/health` — Service health status

> More endpoints will be added as the project evolves.

## 🧪 Development

### Code Quality

Run these commands before committing:

```bash
uv run ruff check --select I --fix   # Fix import sorting
uv run ruff format app/              # Format code
uv run ruff check                    # Full lint check
```

## 🛠️ Technologies

- **FastAPI** — Async web framework
- **Uvicorn** — ASGI server
- **Pydantic / pydantic-settings** — Data validation and configuration
- **itsdangerous** — Secure session cookie signing
- **VenPy** — Interoperability bridge for Powersim `.sip` files *(coming soon)*
- **Gymnasium** — RL environment interface *(coming soon)*
- **Stable-Baselines3** — PPO / SAC agents *(coming soon)*
- **ruff** — Linting and formatting

## 👥 Authors

Natalia Andrea García Ríos
natalia.garcia9@udea.edu.co
ngarciarios2001@gmail.com

Jeisson Alexis Barrantes Toro
jeisson.barrantes@udea.edu.co
jeisson.barrantest@gmail.com

Katherine Rodríguez Mejía
katherine.rodriguezm@udea.edu.co
katherine.road.mei1006@gmail.com