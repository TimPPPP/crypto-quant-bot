.PHONY: install up down run shell test format clean

# --- Setup & Dependencies ---
# Installs dependencies locally (good for your IDE autocompletion)
install:
	pip install poetry
	poetry install

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