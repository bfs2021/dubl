#!/bin/zsh
# Запуск одного шага замера под /usr/bin/time -l; пик памяти (maximum resident set size + peak memory footprint — последний включает память GPU/MPS) дописывается в JSON шага.
# Использование: tests/e2e/rb.sh <имя_json> <шаг> [аргументы...]
cd "$(dirname "$0")/../.." || exit 1
name=$1; shift
mkdir -p ../samples/work/bench
/usr/bin/time -l .venv/bin/python tests/e2e/bench.py "$@" > ../samples/work/bench/$name.out 2> ../samples/work/bench/$name.err
rc=$?
peak=$(grep "maximum resident set size" ../samples/work/bench/$name.err | awk '{print $1}')
foot=$(grep "peak memory footprint" ../samples/work/bench/$name.err | awk '{print $1}')
wall=$(grep " real " ../samples/work/bench/$name.err | awk '{print $1}')
echo "{\"name\":\"$name\",\"rc\":$rc,\"peak_rss_mb\":$(( ${peak:-0} / 1048576 )),\"peak_footprint_mb\":$(( ${foot:-0} / 1048576 )),\"wall\":\"$wall\"}" > ../samples/work/bench/$name.mem.json
cat ../samples/work/bench/$name.mem.json
