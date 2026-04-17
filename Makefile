PUBLIC_IMAGE = ghcr.io/yanghu/immich-ml-router
VENV  = .venv
PY    = $(VENV)/bin/python
PIP   = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest

# Override IMAGE (private registry) in .env.make
-include .env.make

.PHONY: test test-unit test-integration build push push-public release venv

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements-dev.txt
	python3 -m venv $(VENV)
	$(PIP) install -q -r requirements-dev.txt
	touch $(VENV)/bin/activate

test: test-unit test-integration

test-unit: venv
	$(PYTEST) tests/test_router.py -v

test-integration:
	docker compose -f docker-compose.test.yml up -d --build
	bash tests/integration_test.sh
	docker compose -f docker-compose.test.yml down

build:
	docker build -t $(PUBLIC_IMAGE):latest $(if $(IMAGE),-t $(IMAGE):latest,) .

push-public: build
	docker push $(PUBLIC_IMAGE):latest

push: build
ifdef IMAGE
	docker push $(IMAGE):latest
else
	@echo "IMAGE not set in .env.make, skipping private registry push"
endif

release: build push-public push
