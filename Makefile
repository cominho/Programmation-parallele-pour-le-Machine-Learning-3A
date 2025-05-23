.PHONY: deps build clean test

deps:
	pip install -r requirements.txt

build:
	cd att_bench_lib && python build_att_ext.py build_ext --inplace

clean:
	rm -rf build/ dist/ *.egg-info
	rm -f att_bench_lib/*.so att_bench_lib/*.cpp

test:
	pytest -q
