.PHONY: test demo evaluate store-benchmark

test:
	PYTHONPATH=src python -m unittest discover -s test -v

demo:
	PYTHONPATH=src python -m temporal_rag demo --output artifacts/demo.json

evaluate:
	PYTHONPATH=src python -m temporal_rag evaluate --output artifacts/evaluation.json

store-benchmark:
	PYTHONPATH=src python -m temporal_rag store-benchmark --output artifacts/storage-benchmark.json
