.PHONY: test

run:
	uv run main.py

test:
	@set -e; \
	uv run main.py --log-level error --no-access-log > /dev/null 2>&1 & \
	PID=$$!; \
	trap "kill $$PID" EXIT; \
	echo "Running with PID $$PID"; \
	sleep 5; \
	pytest