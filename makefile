.PHONY: test

run:
	uv run main.py

test:
	@set -e; \
	uv run main.py & \
	PID=$$!; \
	trap "kill $$PID" EXIT; \
	echo "Running with PID $$PID"; \
	sleep 5; \
	pytest