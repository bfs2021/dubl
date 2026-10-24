#!/bin/zsh
# Финальная серия замеров (последовательно): e2e на демо-сцене → подгонка → Sintel.
cd "$(dirname "$0")/../.." || exit 1
R=tests/e2e/rb.sh
$R e2e_run e2e best large-v3-turbo
$R fit fit
$R sintel sintel
echo SERIES2_DONE
