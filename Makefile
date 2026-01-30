.PHONY: install up down run shell test format clean env health logs restart setup

# --- Setup & Dependencies ---
# Installs dependencies locally (good for your IDE autocompletion)
install:
	pip install poetry
	poetry install

# Create .env from template
env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example"; \
		echo "⚠ Please edit .env with your actual values"; \
	else \
		echo "⚠ .env already exists, skipping"; \
	fi

# --- Running Your App ---
# Starts the "Permanent" services (Database & API) in the background
# It respects Docker Profiles, so 'ingestion' won't start unless you force it.
up:
	docker-compose up -d

# 2. Dev Mode (Live Logs)
# Starts everything AND immediately follows the logs so you see errors.
dev:
	docker-compose up -d && docker-compose logs -f

# --- The "Universal" Runner (The Generic Command) ---
# Usage: make run cmd="src/encoder.py"
# This spins up a temporary container, runs your script, and deletes the container when done.
run:
	docker-compose run --rm worker python $(cmd)

# Opens a terminal INSIDE the container (useful for debugging)
shell:
	docker-compose run --rm bot /bin/bash

# --- Build ---
build:
	docker-compose build

# --- Maintenance ---
# Stops all containers
down:
	docker-compose down

# Runs tests locally (fastest)
test:
	poetry run pytest tests/

# Formats code locally
format:
	poetry run black src/ tests/

# Deep clean: stops containers and deletes the Database volume (Fresh Start)
clean:
	docker-compose down -v

# Check health of all services
health:
	@echo "Checking service health..."
	@docker-compose ps
	@docker inspect --format='{{.Name}}: {{.State.Health.Status}}' $$(docker-compose ps -q) 2>/dev/null || echo "Health checks not available for all services"

# View logs for all services
logs:
	docker-compose logs -f

# Restart a specific service (usage: make restart service=bot)
restart:
	docker-compose restart $(service)

# Quick setup for new developers
setup: env install build up
	@echo "✓ Setup complete!"
	@echo "Services available at:"
	@echo "  - API/Dashboard: http://localhost:8000"
	@echo "  - QuestDB: http://localhost:9000"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Jupyter: http://localhost:8888"